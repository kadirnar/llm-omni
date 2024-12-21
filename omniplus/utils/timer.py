import torch


def cuda_timer():
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()

    def get_time():
        end.record()
        torch.cuda.synchronize()
        return f"{start.elapsed_time(end) / 1000:.2f}s"

    return get_time
