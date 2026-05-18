/// Avellaneda-Stoikov market-making expert policy.
///
/// Reference: Avellaneda & Stoikov (2008) "High-frequency trading in a
/// limit order book".
///
/// Quotes symmetric half-spread around a reservation price that accounts
/// for inventory risk:
///     r = S + q * gamma * sigma^2 * (T - t)
///     delta = gamma * sigma^2 * (T - t) + (1/gamma) * ln(1 + gamma/k)

/// Avellaneda-Stoikov market-making strategy.
#[derive(Debug, Clone)]
pub struct AvellanedaStoikovExpert {
    /// Risk aversion parameter.
    pub gamma: f64,
    /// Volatility.
    pub sigma: f64,
    /// Time horizon.
    pub T: f64,
    /// Order arrival rate factor.
    pub k: f64,
}

impl AvellanedaStoikovExpert {
    /// Create a new AS expert.
    pub fn new(gamma: f64, sigma: f64, T: f64, k: f64) -> Self {
        Self {
            gamma,
            sigma,
            T,
            k,
        }
    }

    /// Compute reservation price.
    ///
    /// - `mid_price`: current mid-price
    /// - `inventory`: current inventory (positive = long)
    /// - `t`: current time within [0, T]
    pub fn get_reservation_price(&self, mid_price: f64, inventory: f64, t: f64) -> f64 {
        let tau = (self.T - t).max(1e-8);
        mid_price - inventory * self.gamma * self.sigma.powi(2) * tau
    }

    /// Compute optimal half-spread.
    pub fn get_spread(&self, t: f64) -> f64 {
        let tau = (self.T - t).max(1e-8);
        self.gamma * self.sigma.powi(2) * tau
            + (1.0 / self.gamma) * (1.0 + self.gamma / self.k).ln()
    }

    /// Return bid/ask quotes.
    pub fn act(&self, mid_price: f64, inventory: f64, t: f64) -> AsQuotes {
        let r = self.get_reservation_price(mid_price, inventory, t);
        let half_spread = self.get_spread(t);
        AsQuotes {
            bid_price: r - half_spread,
            ask_price: r + half_spread,
            half_spread,
            reservation_price: r,
        }
    }
}

/// Quotes returned by the Avellaneda-Stoikov expert.
#[derive(Debug, Clone)]
pub struct AsQuotes {
    pub bid_price: f64,
    pub ask_price: f64,
    pub half_spread: f64,
    pub reservation_price: f64,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_as_expert() {
        let expert = AvellanedaStoikovExpert::new(0.1, 0.02, 60.0, 1.5);
        let result = expert.act(100.0, 0.0, 10.0);
        assert!(result.bid_price < result.ask_price);
    }

    #[test]
    fn test_as_reservation_price() {
        let expert = AvellanedaStoikovExpert::new(0.1, 0.02, 60.0, 1.5);
        let r = expert.get_reservation_price(100.0, 5.0, 10.0);
        assert!(r < 100.0); // long inventory -> lower reservation
    }
}
