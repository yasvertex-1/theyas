import time
import threading
from collections import deque

class RateLimiter:
    def __init__(self, threshold: int, window_seconds: int):
        self.threshold = threshold
        self.window_seconds = window_seconds
        self.timestamps = deque()
        self.lock = threading.Lock()

    def is_allowed(self) -> bool:
        """
        Check if a request is allowed under the rate limit.
        Removes expired timestamps and checks if the count is within the threshold.
        """
        with self.lock:
            now = time.time()
            # Remove timestamps older than the window
            while self.timestamps and now - self.timestamps[0] > self.window_seconds:
                self.timestamps.popleft()
            
            return len(self.timestamps) < self.threshold

    def record_request(self):
        """
        Record a request timestamp.
        """
        with self.lock:
            self.timestamps.append(time.time())

    def get_current_rate(self) -> int:
        """
        Get the current number of requests in the window.
        """
        with self.lock:
            now = time.time()
            while self.timestamps and now - self.timestamps[0] > self.window_seconds:
                self.timestamps.popleft()
            return len(self.timestamps)
