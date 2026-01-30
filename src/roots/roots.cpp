#include "roots.hpp"
#include <cmath>
#include <stdexcept>
#include <functional>

// ---------------------------------------------------------------------------
// Simple root-finding utilities implementing four common methods:
//  - bisection
//  - regula falsi (false position)
//  - newton-raphson
//  - secant
// The functions are intentionally simple: they accept a callable `f`, an
// interval `[a,b]`, and a `double* root` to store the result. Helper
// validation and small utility functions live in the anonymous namespace.
// ---------------------------------------------------------------------------

namespace {
    // upper bounds for iterations and the convergence tolerance used across
    // the algorithms in this file.
    constexpr int MAX_ITERATIONS = 1'000'000;
    constexpr double TOLERANCE = 1e-6;

    // Return true when a value is close enough to zero (absolute tolerance).
    bool is_effectively_zero(double v, double abs_tol = TOLERANCE) {
        return std::abs(v) <= abs_tol;
    }

    // Input validation helpers that throw a descriptive exception on error.
    void validate_root_ptr(double* root) {
        if (!root) throw std::invalid_argument("Root pointer cannot be null");
    }

    void validate_interval(double a, double b) {
        if (a >= b) throw std::invalid_argument("a must be less than b");
    }

    void validate_c_in_interval(double a, double b, double c) {
        if (c < a || c > b) throw std::invalid_argument("c must be within [a, b]");
    }
}

// ---------------------------------------------------------------------------
// Bisection method
// - `f` : function to find root of
// - `[a,b]` : interval that must bracket a root (f(a) and f(b) opposite sign)
// - `root` : output location for the computed root
// Returns true on success and writes the root to `*root`.
// The implementation repeatedly halves the interval until the function at
// the midpoint is near zero or the interval width falls below tolerance.
// ---------------------------------------------------------------------------
bool bisection(std::function<double(double)> f, double a, double b, double* root) {
    validate_root_ptr(root);
    validate_interval(a, b);

    double fa = f(a);
    double fb = f(b);

    // If an endpoint is already (approximately) a root, return it immediately.
    if (is_effectively_zero(fa)) { *root = a; return true; }
    if (is_effectively_zero(fb)) { *root = b; return true; }

    // Require the interval to bracket a sign change.
    if (fa * fb > 0) return false;

    double left = a, right = b;
    double fleft = fa;

    for (int i = 0; i < MAX_ITERATIONS; ++i) {
        double mid  = 0.5 * (left + right);
        double fmid = f(mid);

        // Convergence: function value near zero
        if (is_effectively_zero(fmid)) {
            *root = mid;
            return true;
        }

        // Convergence: interval is sufficiently small
        if ((right - left) <= TOLERANCE) {
            *root = mid;
            return true;
        }

        // Keep the sub-interval that brackets the sign change.
        if (fleft * fmid <= 0) {
            right = mid;
        } else {
            left = mid;
            fleft = fmid;
        }
    }

    // If we exhaust iterations, return the midpoint (behaves like original)
    *root = 0.5 * (left + right);
    return true;
}

// ---------------------------------------------------------------------------
// Regula Falsi (False Position)
// Similar to bisection but uses the intersection of the secant through the
// endpoints with the x-axis as the next approximation. If the computed
// denominator would be (near) zero, the implementation falls back to a
// bisection-like midpoint step to avoid division by a tiny number.
// ---------------------------------------------------------------------------
bool regula_falsi(std::function<double(double)> f, double a, double b, double* root) {
    validate_root_ptr(root);
    validate_interval(a, b);

    double fa = f(a);
    double fb = f(b);

    if (is_effectively_zero(fa)) { *root = a; return true; }
    if (is_effectively_zero(fb)) { *root = b; return true; }

    if (fa * fb > 0) return false;

    double x0 = a, x1 = b;
    double fx0 = fa, fx1 = fb;

    for (int i = 0; i < MAX_ITERATIONS; ++i) {
        double denom = fx1 - fx0;

        // If denominator is too small, avoid the false-position formula and
        // perform a safe bisection step instead.
        if (is_effectively_zero(denom, TOLERANCE)) {
            double mid = 0.5 * (x0 + x1);
            double fmid = f(mid);

            if (is_effectively_zero(fmid)) {
                *root = mid;
                return true;
            }

            if (fx0 * fmid <= 0) {
                x1 = mid; fx1 = fmid;
            } else {
                x0 = mid; fx0 = fmid;
            }
            continue;
        }

        // False position formula: intersection of secant (x0,fx0)-(x1,fx1)
        double x_new = (x0 * fx1 - x1 * fx0) / denom;
        double f_new = f(x_new);

        if (is_effectively_zero(f_new)) {
            *root = x_new;
            return true;
        }

        if (std::abs(x1 - x0) <= TOLERANCE || is_effectively_zero(f_new)) {
            *root = x_new;
            return true;
        }

        // Update bracketing interval to maintain a sign change.
        if (fx0 * f_new < 0) {
            x1 = x_new; fx1 = f_new;
        } else {
            x0 = x_new; fx0 = f_new;
        }
    }

    *root = 0.5 * (x0 + x1);
    return is_effectively_zero(f(*root));
}

// ---------------------------------------------------------------------------
// Newton-Raphson
// Fast, derivative-based method: requires `g` = f'(x). Fails if derivative
// is (nearly) zero or if the Newton step leaves the interval [a,b].
// ---------------------------------------------------------------------------
bool newton_raphson(std::function<double(double)> f,
                    std::function<double(double)> g,
                    double a, double b, double c,
                    double* root) {
    validate_root_ptr(root);
    validate_interval(a, b);
    validate_c_in_interval(a, b, c);

    double x = c;
    double fx = f(x);

    if (is_effectively_zero(fx)) { *root = x; return true; }

    for (int i = 0; i < MAX_ITERATIONS; ++i) {
        double gx = g(x);

        // Avoid division by near-zero derivative.
        if (is_effectively_zero(gx, TOLERANCE)) {
            return false;
        }

        double x_new = x - fx / gx;

        // Keep iterates inside the provided interval (consistent with original)
        if (x_new < a || x_new > b) return false;

        double f_new = f(x_new);

        if (is_effectively_zero(f_new)) {
            *root = x_new;
            return true;
        }

        // Small change in x combined with small f_new indicates convergence.
        double x_change = std::abs(x_new - x);
        double x_scale  = std::max(std::abs(x_new), std::abs(x));
        if (x_change <= TOLERANCE + TOLERANCE * x_scale && is_effectively_zero(f_new)) {
            *root = x_new;
            return true;
        }

        x = x_new;
        fx = f_new;
    }

    // If iterations exhausted, return last iterate and indicate success only
    // if the function value is close to zero.
    *root = x;
    return is_effectively_zero(fx);
}

// ---------------------------------------------------------------------------
// Secant method
// Like Newton but approximates derivative using two recent points. The
// implementation selects a second starting point near `c` and clamps steps
// into [a,b]. It also guards against tiny denominators.
// ---------------------------------------------------------------------------
bool secant(std::function<double(double)> f, double a, double b, double c, double* root) {
    validate_root_ptr(root);
    validate_interval(a, b);
    validate_c_in_interval(a, b, c);

    double x0 = c;
    double fx0 = f(x0);

    if (is_effectively_zero(fx0)) { *root = x0; return true; }

    // Choose a second point x1 near c while preferring an endpoint if it
    // helps preserve a bracketing sign change.
    double x1;
    if (c == a) {
        x1 = a + (b - a) * 0.1;
    } else if (c == b) {
        x1 = b - (b - a) * 0.1;
    } else {
        double fa = f(a);
        double fb = f(b);

        if (fa * fx0 < 0) {
            x1 = a;
        } else if (fb * fx0 < 0) {
            x1 = b;
        } else {
            double dx = (b - a) * 0.1;
            x1 = c + dx;
            if (x1 > b) x1 = c - dx;
        }
    }

    // Clamp x1 into the interval bounds and ensure it's different from x0.
    if (x1 < a) x1 = a;
    if (x1 > b) x1 = b;

    if (is_effectively_zero(x1 - x0, TOLERANCE)) return false;

    double fx1 = f(x1);
    if (is_effectively_zero(fx1)) { *root = x1; return true; }

    for (int i = 0; i < MAX_ITERATIONS; ++i) {
        double denom = fx1 - fx0;
        if (is_effectively_zero(denom, TOLERANCE)) return false;

        // Secant update (approximate Newton step without analytic derivative)
        double x_new = x1 - fx1 * (x1 - x0) / denom;

        // Clamp the new point into [a,b] to match original behaviour.
        if (x_new < a) x_new = a;
        if (x_new > b) x_new = b;

        double f_new = f(x_new);

        if (is_effectively_zero(f_new)) {
            *root = x_new;
            return true;
        }

        double x_change = std::abs(x_new - x1);
        double x_scale  = std::max(std::abs(x_new), std::abs(x1));
        if (x_change <= TOLERANCE + TOLERANCE * x_scale) {
            *root = x_new;
            return is_effectively_zero(f_new);
        }

        // Advance the secant pair for the next iteration.
        x0 = x1; fx0 = fx1;
        x1 = x_new; fx1 = f_new;
    }

    *root = x1;
    return is_effectively_zero(fx1);
}
