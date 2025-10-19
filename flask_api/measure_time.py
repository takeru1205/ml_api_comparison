import time
import statistics as stats
import requests
from typing import Dict, Any, Iterable, Tuple


def run_benchmark(
    url: str,
    n_requests: int = 100,
    timeout: float = 10.0,
    payload: Dict[str, Any] | None = None,
    headers: Dict[str, str] | None = None,
    quiet: bool = False,
) -> Dict[str, Any]:
    """
    指定URLに同一ペイロードで n_requests 回 POST し、各リクエスト時間を計測して統計を返す。
    失敗（例外・非2xx）も1試行とみなし、経過時間は集計に含める。
    """
    if payload is None:
        payload = {
            "Pclass": 0,
            "Sex": "male",
            "Age": 0,
            "SibSp": 0,
            "Parch": 0,
            "Fare": 0,
        }
    if headers is None:
        headers = {"accept": "application/json", "Content-Type": "application/json"}

    times: list[float] = []
    success = 0
    failures = 0
    first_response = None
    status_codes: list[int] = []

    with requests.Session() as sess:
        for i in range(n_requests):
            start = time.perf_counter()
            try:
                resp = sess.post(url, json=payload, headers=headers, timeout=timeout)
                elapsed = time.perf_counter() - start
                times.append(elapsed)
                status_codes.append(resp.status_code)

                if 200 <= resp.status_code < 300:
                    success += 1
                    if first_response is None:
                        try:
                            first_response = resp.json()
                        except Exception:
                            first_response = resp.text
                else:
                    failures += 1
            except Exception:
                elapsed = time.perf_counter() - start
                times.append(elapsed)
                status_codes.append(-1)  # 例外を -1 として記録
                failures += 1

    # 統計量
    total = sum(times)
    avg = total / len(times) if times else float("nan")
    pvar = stats.pvariance(times) if len(times) > 0 else float("nan")   # 母分散
    svar = stats.variance(times) if len(times) > 1 else float("nan")    # 標本分散
    t_min = min(times) if times else float("nan")
    t_max = max(times) if times else float("nan")

    # 表示用（ms）
    to_ms = lambda s: f"{s*1000:.3f} ms"

    if not quiet:
        print("=== Request Benchmark ===")
        print(f"URL: {url}")
        print(f"Requests: {n_requests} (success: {success}, failures: {failures})")
        print(f"Total time: {to_ms(total)}")
        print(f"Average:    {to_ms(avg)}")
        print(f"Min:        {to_ms(t_min)}")
        print(f"Max:        {to_ms(t_max)}")
        print(f"Pop Var:    {(pvar*1_000_000):.3f} (ms^2)")
        print(f"Sample Var: {(svar*1_000_000):.3f} (ms^2)")
        if first_response is not None:
            print("\n--- Sample response (first successful) ---")
            print(first_response)

    # 返り値（サマリー用）
    return {
        "url": url,
        "requests": n_requests,
        "success": success,
        "failures": failures,
        "status_codes": status_codes,
        "total_s": total,
        "avg_s": avg,
        "min_s": t_min,
        "max_s": t_max,
        "pop_var_s2": pvar,
        "sample_var_s2": svar,
        "total_ms": total * 1000,
        "avg_ms": avg * 1000,
        "min_ms": t_min * 1000,
        "max_ms": t_max * 1000,
        "pop_var_ms2": pvar * 1_000_000,
        "sample_var_ms2": svar * 1_000_000,
        "sample_response": first_response,
    }


def build_urls(base: str, paths: Iterable[str]) -> Iterable[Tuple[str, str]]:
    """
    base と paths から (path, url) のタプルを返す。
    """
    base = base.rstrip("/")
    for p in paths:
        if not p.startswith("/"):
            p = "/" + p
        yield p, base + p


def run_all(
    base: str = "http://localhost:8000",
    paths: Iterable[str] = ("/predict/torch/", "/predict/lgb/", "/predict/torch_onnx/", "/predict/lgb_onnx/"),
    n_requests: int = 100,
    timeout: float = 10.0,
) -> Dict[str, Dict[str, Any]]:
    """
    複数エンドポイントを順次ベンチマークし、結果辞書を返す。
    """
    results: Dict[str, Dict[str, Any]] = {}
    for path, url in build_urls(base, paths):
        print(f"\n### Benchmarking {url}")
        res = run_benchmark(url=url, n_requests=n_requests, timeout=timeout, quiet=False)
        results[path] = res

    # サマリ（CSVライク、単位は ms）
    print("\n=== Summary (ms) ===")
    print("endpoint,avg,min,max,total,pop_var,sample_var,success,failures")
    for p, s in results.items():
        print(
            f"{p},{s['avg_ms']:.3f},{s['min_ms']:.3f},{s['max_ms']:.3f},"
            f"{s['total_ms']:.3f},{s['pop_var_ms2']:.3f},{s['sample_var_ms2']:.3f},"
            f"{s['success']},{s['failures']}"
        )

    return results


if __name__ == "__main__":
    # 必要に応じて base / n_requests / timeout を調整してください
    run_all(
        base="http://localhost:8000",
        paths=(
            "/predict/torch/",
            "/predict/lgb/",
            "/predict/torch_onnx/",
            "/predict/lgb_onnx/",
        ),
        n_requests=1000,
        timeout=10.0,
    )

