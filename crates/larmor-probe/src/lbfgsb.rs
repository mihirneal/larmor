//! scipy's L-BFGS-B (`scipy/optimize/src/lbfgsb.c`, scipy 1.18.1) for problems without bounds, which is the
//! solver behind `LogisticRegressionCV(solver="lbfgs")`.
//!
//! sklearn stops L-BFGS-B at a gradient tolerance of 1e-4, well short of the optimum, so matching its
//! coefficients means retracing its iterates: the same Cauchy step on the first iteration, the same
//! Moré-Thuente line search (`dcsrch`/`dcstep`), the same memory updates and stopping tests. Without bounds
//! the subspace minimisation reduces to the quasi-Newton step `-H g`, computed here with the two-loop
//! recursion instead of the compact form, which agrees to rounding.

use std::collections::VecDeque;

/// Options for [`minimize`], named as in `scipy.optimize.minimize(method="L-BFGS-B")`.
#[derive(Clone, Copy, Debug)]
pub struct LbfgsbOptions {
    /// History size (`maxcor`).
    pub memory: usize,
    /// Stop when the largest gradient component is at most this (`gtol`, `pgtol`).
    pub gtol: f64,
    /// Stop when the relative decrease of f is at most this (`ftol` = `factr * eps`).
    pub ftol: f64,
    /// Maximum number of iterations (`maxiter`).
    pub max_iter: usize,
    /// Maximum number of line-search steps per iteration (`maxls`).
    pub max_line_search: usize,
}

/// Why [`minimize`] stopped.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Stop {
    /// The largest gradient component fell to `gtol`.
    Gradient,
    /// The relative decrease of f fell to `ftol`.
    Function,
    /// `max_iter` iterations ran.
    MaxIter,
    /// The line search failed with an empty memory (`ABNORMAL`).
    Abnormal,
    /// The objective or its gradient became NaN or infinite; the point is not a minimum.
    NonFinite,
}

/// The point where [`minimize`] stopped.
#[derive(Clone, Debug)]
pub struct Minimum {
    pub x: Vec<f64>,
    pub f: f64,
    /// Iterations run (`OptimizeResult.nit`).
    pub n_iter: usize,
    pub stop: Stop,
}

/// Minimises `f` from `x0`; `objective` returns the value and the gradient.
pub fn minimize(
    mut objective: impl FnMut(&[f64]) -> (f64, Vec<f64>),
    x0: &[f64],
    options: &LbfgsbOptions,
) -> Minimum {
    let mut x = x0.to_vec();
    let (mut f, mut g) = objective(&x);
    if !is_finite(f, &g) {
        return Minimum { x, f, n_iter: 0, stop: Stop::NonFinite };
    }
    if max_abs(&g) <= options.gtol {
        return Minimum { x, f, n_iter: 0, stop: Stop::Gradient };
    }
    let mut memory = Memory::new(options.memory);
    let mut n_iter = 0;
    loop {
        // the trial point z: a steepest-descent Cauchy step with an empty memory, else the quasi-Newton step
        let z: Vec<f64> = if memory.is_empty() {
            x.iter().zip(&g).map(|(xi, gi)| xi + memory.cauchy_step() * -gi).collect()
        } else {
            let step = memory.apply_inverse_hessian(&g);
            x.iter().zip(&step).map(|(xi, si)| xi - si).collect()
        };
        let mut d: Vec<f64> = z.iter().zip(&x).map(|(zi, xi)| zi - xi).collect();

        let start = LineStart { x: x.clone(), f, g: g.clone() };
        let first_iteration = n_iter == 0;
        let search = line_search(&mut objective, &start, &z, &d, first_iteration, options.max_line_search);
        let Some(accepted) = search else {
            x = start.x;
            g = start.g;
            f = start.f;
            if memory.is_empty() {
                return Minimum { x, f, n_iter: n_iter + 1, stop: Stop::Abnormal };
            }
            memory.clear();
            continue;
        };
        x = accepted.x;
        f = accepted.f;
        g = accepted.g;
        n_iter += 1;
        if !is_finite(f, &g) {
            return Minimum { x, f, n_iter, stop: Stop::NonFinite };
        }

        // scipy's Python driver checks maxiter before the convergence tests
        if n_iter >= options.max_iter {
            return Minimum { x, f, n_iter, stop: Stop::MaxIter };
        }
        if max_abs(&g) <= options.gtol {
            return Minimum { x, f, n_iter, stop: Stop::Gradient };
        }
        let scale = start.f.abs().max(f.abs()).max(1.0);
        if start.f - f <= options.ftol * scale {
            return Minimum { x, f, n_iter, stop: Stop::Function };
        }

        // the correction pair, skipped unless the curvature s'y is clearly positive
        let y: Vec<f64> = g.iter().zip(&start.g).map(|(new, old)| new - old).collect();
        let (s_dot_y, reference) = if accepted.step == 1.0 {
            (accepted.gd - accepted.gd_initial, -accepted.gd_initial)
        } else {
            for di in &mut d {
                *di *= accepted.step;
            }
            ((accepted.gd - accepted.gd_initial) * accepted.step, -accepted.gd_initial * accepted.step)
        };
        if s_dot_y <= f64::EPSILON * reference {
            continue;
        }
        let y_dot_y = dot(&y, &y);
        memory.push(d, y, s_dot_y, y_dot_y);
    }
}

struct LineStart {
    x: Vec<f64>,
    f: f64,
    g: Vec<f64>,
}

struct Accepted {
    x: Vec<f64>,
    f: f64,
    g: Vec<f64>,
    step: f64,
    /// `g' d` at the accepted point and at the start.
    gd: f64,
    gd_initial: f64,
}

/// `lnsrlb` without bounds: Moré-Thuente on `x + stp d`, starting at `1/||d||` on the first iteration and at
/// 1 afterwards. `None` is a failed search (uphill direction, or `max_line_search` trial points).
fn line_search(
    objective: &mut impl FnMut(&[f64]) -> (f64, Vec<f64>),
    start: &LineStart,
    z: &[f64],
    d: &[f64],
    first_iteration: bool,
    max_line_search: usize,
) -> Option<Accepted> {
    const STEP_MAX: f64 = 1e10;
    let d_norm = dot(d, d).sqrt();
    let mut step = if first_iteration { (1.0 / d_norm).min(STEP_MAX) } else { 1.0 };
    let gd_initial = dot(&start.g, d);
    if gd_initial >= 0.0 {
        return None;
    }
    let mut search = Dcsrch::new(start.f, gd_initial, step, 1e-3, 0.9, 0.1, 0.0, STEP_MAX);
    let mut x = start.x.clone();
    let (mut f, mut g, mut gd) = (start.f, start.g.clone(), gd_initial);
    let mut trials = 0;
    loop {
        match search.next(f, gd, &mut step) {
            Search::Evaluate => {
                trials += 1;
                if trials > max_line_search {
                    return None;
                }
                // at step 1 scipy takes the trial point itself rather than x + 1 * d
                x = if step == 1.0 {
                    z.to_vec()
                } else {
                    d.iter().zip(&start.x).map(|(di, xi)| step * di + xi).collect()
                };
                (f, g) = objective(&x);
                gd = dot(&g, d);
            }
            Search::Done => return Some(Accepted { x, f, g, step, gd, gd_initial }),
        }
    }
}

enum Search {
    Evaluate,
    Done,
}

/// MINPACK-2 `dcsrch`: the Moré-Thuente line search, as a state machine fed one (f, g'd) at a time.
struct Dcsrch {
    gtol: f64,
    xtol: f64,
    step_min: f64,
    step_max: f64,
    started: bool,
    bracketed: bool,
    stage: u8,
    f_init: f64,
    g_init: f64,
    g_test: f64,
    width: f64,
    width1: f64,
    stx: f64,
    fx: f64,
    gx: f64,
    sty: f64,
    fy: f64,
    gy: f64,
    stmin: f64,
    stmax: f64,
}

impl Dcsrch {
    #[allow(clippy::too_many_arguments)]
    fn new(f: f64, g: f64, step: f64, ftol: f64, gtol: f64, xtol: f64, step_min: f64, step_max: f64) -> Self {
        let width = step_max - step_min;
        Self {
            gtol,
            xtol,
            step_min,
            step_max,
            started: false,
            bracketed: false,
            stage: 1,
            f_init: f,
            g_init: g,
            g_test: ftol * g,
            width,
            width1: width / 0.5,
            stx: 0.0,
            fx: f,
            gx: g,
            sty: 0.0,
            fy: f,
            gy: g,
            stmin: 0.0,
            stmax: step + 4.0 * step,
        }
    }

    /// One call of `dcsrch` with the function value and directional derivative at the current `step`.
    fn next(&mut self, f: f64, g: f64, step: &mut f64) -> Search {
        if !self.started {
            // task == START: the first call only asks for the first trial point
            self.started = true;
            return Search::Evaluate;
        }
        let f_test = self.f_init + *step * self.g_test;
        if self.stage == 1 && f <= f_test && g >= 0.0 {
            self.stage = 2;
        }
        let warning = (self.bracketed && (*step <= self.stmin || *step >= self.stmax))
            || (self.bracketed && self.stmax - self.stmin <= self.xtol * self.stmax)
            || (*step == self.step_max && f <= f_test && g <= self.g_test)
            || (*step == self.step_min && (f > f_test || g >= self.g_test));
        let converged = f <= f_test && g.abs() <= self.gtol * -self.g_init;
        if warning || converged {
            return Search::Done;
        }

        // in the first stage, with a sufficient-decrease failure still possible, work on the modified
        // function
        if self.stage == 1 && f <= self.fx && f > f_test {
            let fm = f - *step * self.g_test;
            let mut fxm = self.fx - self.stx * self.g_test;
            let mut fym = self.fy - self.sty * self.g_test;
            let gm = g - self.g_test;
            let mut gxm = self.gx - self.g_test;
            let mut gym = self.gy - self.g_test;
            dcstep(
                &mut self.stx,
                &mut fxm,
                &mut gxm,
                &mut self.sty,
                &mut fym,
                &mut gym,
                step,
                fm,
                gm,
                &mut self.bracketed,
                self.stmin,
                self.stmax,
            );
            self.fx = fxm + self.stx * self.g_test;
            self.fy = fym + self.sty * self.g_test;
            self.gx = gxm + self.g_test;
            self.gy = gym + self.g_test;
        } else {
            dcstep(
                &mut self.stx,
                &mut self.fx,
                &mut self.gx,
                &mut self.sty,
                &mut self.fy,
                &mut self.gy,
                step,
                f,
                g,
                &mut self.bracketed,
                self.stmin,
                self.stmax,
            );
        }

        if self.bracketed {
            if (self.sty - self.stx).abs() >= 0.66 * self.width1 {
                *step = self.stx + 0.5 * (self.sty - self.stx);
            }
            self.width1 = self.width;
            self.width = (self.sty - self.stx).abs();
            self.stmin = self.stx.min(self.sty);
            self.stmax = self.stx.max(self.sty);
        } else {
            self.stmin = *step + 1.1 * (*step - self.stx);
            self.stmax = *step + 4.0 * (*step - self.stx);
        }
        *step = step.max(self.step_min).min(self.step_max);
        if (self.bracketed && (*step <= self.stmin || *step >= self.stmax))
            || (self.bracketed && self.stmax - self.stmin <= self.xtol * self.stmax)
        {
            *step = self.stx;
        }
        Search::Evaluate
    }
}

/// MINPACK-2 `dcstep`: a safeguarded cubic or quadratic step and the update of the bracketing interval.
#[allow(clippy::too_many_arguments)]
fn dcstep(
    stx: &mut f64,
    fx: &mut f64,
    dx: &mut f64,
    sty: &mut f64,
    fy: &mut f64,
    dy: &mut f64,
    stp: &mut f64,
    fp: f64,
    dp: f64,
    bracketed: &mut bool,
    stpmin: f64,
    stpmax: f64,
) {
    let sgnd = dp * (*dx / dx.abs());
    let stpf;
    if fp > *fx {
        // higher function value: the minimum is bracketed
        let theta = 3.0 * (*fx - fp) / (*stp - *stx) + *dx + dp;
        let s = theta.abs().max(dx.abs()).max(dp.abs());
        let mut gamma = s * ((theta / s).powi(2) - (*dx / s) * (dp / s)).sqrt();
        if *stp < *stx {
            gamma = -gamma;
        }
        let p = (gamma - *dx) + theta;
        let q = ((gamma - *dx) + gamma) + dp;
        let r = p / q;
        let stpc = *stx + r * (*stp - *stx);
        let stpq = *stx + ((*dx / ((*fx - fp) / (*stp - *stx) + *dx)) / 2.0) * (*stp - *stx);
        stpf = if (stpc - *stx).abs() < (stpq - *stx).abs() { stpc } else { stpc + (stpq - stpc) / 2.0 };
        *bracketed = true;
    } else if sgnd < 0.0 {
        // derivatives of opposite sign: bracketed
        let theta = 3.0 * (*fx - fp) / (*stp - *stx) + *dx + dp;
        let s = theta.abs().max(dx.abs()).max(dp.abs());
        let mut gamma = s * ((theta / s).powi(2) - (*dx / s) * (dp / s)).sqrt();
        if *stp > *stx {
            gamma = -gamma;
        }
        let p = (gamma - dp) + theta;
        let q = ((gamma - dp) + gamma) + *dx;
        let r = p / q;
        let stpc = *stp + r * (*stx - *stp);
        let stpq = *stp + (dp / (dp - *dx)) * (*stx - *stp);
        stpf = if (stpc - *stp).abs() > (stpq - *stp).abs() { stpc } else { stpq };
        *bracketed = true;
    } else if dp.abs() < dx.abs() {
        // same sign, the derivative magnitude decreases
        let theta = 3.0 * (*fx - fp) / (*stp - *stx) + *dx + dp;
        let s = theta.abs().max(dx.abs()).max(dp.abs());
        let mut gamma = s * (0.0f64).max((theta / s).powi(2) - (*dx / s) * (dp / s)).sqrt();
        if *stp > *stx {
            gamma = -gamma;
        }
        let p = (gamma - dp) + theta;
        let q = (gamma + (*dx - dp)) + gamma;
        let r = p / q;
        let stpc = if r < 0.0 && gamma != 0.0 {
            *stp + r * (*stx - *stp)
        } else if *stp > *stx {
            stpmax
        } else {
            stpmin
        };
        let stpq = *stp + (dp / (dp - *dx)) * (*stx - *stp);
        if *bracketed {
            let closer = if (stpc - *stp).abs() < (stpq - *stp).abs() { stpc } else { stpq };
            stpf = if *stp > *stx {
                (*stp + 0.66 * (*sty - *stp)).min(closer)
            } else {
                (*stp + 0.66 * (*sty - *stp)).max(closer)
            };
        } else {
            let farther = if (stpc - *stp).abs() > (stpq - *stp).abs() { stpc } else { stpq };
            stpf = farther.min(stpmax).max(stpmin);
        }
    } else if *bracketed {
        // same sign, the derivative magnitude does not decrease
        let theta = 3.0 * (fp - *fy) / (*sty - *stp) + *dy + dp;
        let s = theta.abs().max(dy.abs()).max(dp.abs());
        let mut gamma = s * ((theta / s).powi(2) - (*dy / s) * (dp / s)).sqrt();
        if *stp > *sty {
            gamma = -gamma;
        }
        let p = (gamma - dp) + theta;
        let q = ((gamma - dp) + gamma) + *dy;
        let r = p / q;
        stpf = *stp + r * (*sty - *stp);
    } else if *stp > *stx {
        stpf = stpmax;
    } else {
        stpf = stpmin;
    }

    if fp > *fx {
        *sty = *stp;
        *fy = fp;
        *dy = dp;
    } else {
        if sgnd < 0.0 {
            *sty = *stx;
            *fy = *fx;
            *dy = *dx;
        }
        *stx = *stp;
        *fx = fp;
        *dx = dp;
    }
    *stp = stpf;
}

/// The limited memory: the last `capacity` pairs (s, y) and `theta = y'y / s'y` of the newest pair.
struct Memory {
    capacity: usize,
    pairs: VecDeque<(Vec<f64>, Vec<f64>, f64)>,
    theta: f64,
}

impl Memory {
    fn new(capacity: usize) -> Self {
        Self { capacity, pairs: VecDeque::with_capacity(capacity), theta: 1.0 }
    }

    fn is_empty(&self) -> bool {
        self.pairs.is_empty()
    }

    fn clear(&mut self) {
        self.pairs.clear();
        self.theta = 1.0;
    }

    /// `cauchy` with an empty memory: the quadratic model `theta I` along `-g` is minimised at `1 / theta`.
    fn cauchy_step(&self) -> f64 {
        1.0 / self.theta
    }

    fn push(&mut self, s: Vec<f64>, y: Vec<f64>, s_dot_y: f64, y_dot_y: f64) {
        if self.pairs.len() == self.capacity {
            self.pairs.pop_front();
        }
        self.pairs.push_back((s, y, 1.0 / s_dot_y));
        self.theta = y_dot_y / s_dot_y;
    }

    /// `H g` by the two-loop recursion with `H0 = I / theta`.
    fn apply_inverse_hessian(&self, g: &[f64]) -> Vec<f64> {
        let mut q = g.to_vec();
        let mut alphas = vec![0.0; self.pairs.len()];
        for (i, (s, y, rho)) in self.pairs.iter().enumerate().rev() {
            alphas[i] = rho * dot(s, &q);
            axpy(-alphas[i], y, &mut q);
        }
        let mut r: Vec<f64> = q.iter().map(|v| v / self.theta).collect();
        for (i, (s, y, rho)) in self.pairs.iter().enumerate() {
            let beta = rho * dot(y, &r);
            axpy(alphas[i] - beta, s, &mut r);
        }
        r
    }
}

fn dot(a: &[f64], b: &[f64]) -> f64 {
    a.iter().zip(b).map(|(x, y)| x * y).sum()
}

fn axpy(a: f64, x: &[f64], y: &mut [f64]) {
    for (yi, xi) in y.iter_mut().zip(x) {
        *yi += a * xi;
    }
}

/// The largest absolute value, NaN if any value is NaN (`f64::max` alone would drop a NaN, and a NaN
/// gradient would then pass for a converged one).
fn max_abs(values: &[f64]) -> f64 {
    values.iter().fold(0.0, |m: f64, v| if m.is_nan() || v.is_nan() { f64::NAN } else { m.max(v.abs()) })
}

fn is_finite(f: f64, g: &[f64]) -> bool {
    f.is_finite() && g.iter().all(|v| v.is_finite())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_nan_gradient_is_not_convergence() {
        assert!(max_abs(&[1.0, f64::NAN, 2.0]).is_nan());
        assert_eq!(max_abs(&[-3.0, 2.0]), 3.0);
        let options =
            LbfgsbOptions { memory: 10, gtol: 1e-4, ftol: 1e-12, max_iter: 100, max_line_search: 20 };
        let nan = minimize(|x| (f64::NAN, vec![f64::NAN; x.len()]), &[1.0, 2.0], &options);
        assert_eq!(nan.stop, Stop::NonFinite);
        // finite at the start, NaN once x leaves the unit ball
        let blows_up = minimize(
            |x| {
                let r2: f64 = x.iter().map(|v| v * v).sum();
                if r2 > 1.0 { (f64::NAN, vec![f64::NAN; 2]) } else { (-x[0], vec![-1.0, 0.0]) }
            },
            &[0.0, 0.0],
            &options,
        );
        assert_ne!(blows_up.stop, Stop::Gradient);
    }

    #[test]
    fn minimises_the_rosenbrock_function() {
        let rosenbrock = |x: &[f64]| {
            let (a, b) = (x[0], x[1]);
            let f = (1.0 - a).powi(2) + 100.0 * (b - a * a).powi(2);
            let g = vec![-2.0 * (1.0 - a) - 400.0 * a * (b - a * a), 200.0 * (b - a * a)];
            (f, g)
        };
        let options =
            LbfgsbOptions { memory: 10, gtol: 1e-10, ftol: 0.0, max_iter: 1000, max_line_search: 50 };
        let minimum = minimize(rosenbrock, &[-1.2, 1.0], &options);
        assert!((minimum.x[0] - 1.0).abs() < 1e-6 && (minimum.x[1] - 1.0).abs() < 1e-6, "{minimum:?}");
    }
}
