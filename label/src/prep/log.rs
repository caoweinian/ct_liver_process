use std::time::{Duration, Instant};

/// benchmark计时结构。
#[derive(Clone)]
pub struct AccTimer {
    consumed: Duration,
    since: Instant,
}

impl AccTimer {
    /// 初始化计时器。初始化时会视为已经调用一次`self.start()`。如果用户不希望这种行为，可以在需要时重新调用`self.start()`来覆盖该行为。
    #[inline]
    pub fn new() -> Self {
        Self {
            consumed: Duration::from_micros(0),
            since: Instant::now(),
        }
    }

    /// 开始计时。可以通过再次调用来重置，或者通过之后的`self.elapsed()`方法来统计该部分时间。
    #[inline]
    pub fn start(&mut self) {
        self.since = Instant::now();
    }

    /// 结束计时，并将这一区间的时间累计起来。上一次调用必须是`self.start()`，否则时间计算值无意义。
    #[inline]
    pub fn elapsed(&mut self) {
        self.consumed += self.since.elapsed();
    }

    /// 获得总共累计下来的时间综合（以微秒为单位）。
    #[inline]
    pub fn get_total_us(&self) -> u64 {
        self.consumed.as_micros() as u64
    }

    /// 获得总共累计下来的时间综合（以毫秒为单位）。
    #[inline]
    pub fn get_total_ms(&self) -> u64 {
        self.consumed.as_millis() as u64
    }
}

impl Default for AccTimer {
    fn default() -> Self {
        Self::new()
    }
}
