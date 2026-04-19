import threading
from delta_rich import Stats

def test_stats_thread_safety():
    stats = Stats()
    def increment():
        for _ in range(1000):
            stats.increment("copied")
    
    threads = [threading.Thread(target=increment) for _ in range(10)]
    for t in threads: t.start()
    for t in threads: t.join()
    assert stats.get("copied") == 10000
