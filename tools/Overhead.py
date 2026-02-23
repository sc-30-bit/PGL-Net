import torch
import torch.nn as nn
import os
import sys

def count_parameters(model):
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total_params, trainable_params

def format_params(num):
    return f"{num / 1_000_000:.2f} million"

def count_macs(model, input_size=(1, 3, 256, 256)):
    try:
        from torchprofile import profile_macs
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = model.to(device)
        x = torch.randn(input_size).to(device)
        with torch.no_grad():
            macs = profile_macs(model, x)
        return macs
    except ImportError:
        try:
            from thop import profile
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            model = model.to(device)
            x = torch.randn(input_size).to(device)
            with torch.no_grad():
                macs, _ = profile(model, inputs=(x,))
            return macs
        except ImportError:
            print("Warning: torchprofile or thop library not installed, unable to calculate MACs")
            return None

def format_macs(num):
    if num is None:
        return "Unable to calculate"
    if num >= 1e12:
        return f"{num / 1e12:.2f} TMACs"
    elif num >= 1e9:
        return f"{num / 1e9:.2f} GMACs"
    elif num >= 1e6:
        return f"{num / 1e6:.2f} MMACs"
    else:
        return f"{num:.2f} MACs"


def calculate_model_overhead(model, model_name="Model", input_size=(1, 3, 256, 256)):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    model.eval()

    total_params, trainable_params = count_parameters(model)
    macs = count_macs(model, input_size)

    print(f"\n{'='*50}")
    print(f"Model: {model_name}")
    print(f"{'='*50}")
    print(f"Total parameters: {format_params(total_params)}")
    print(f"Trainable parameters: {format_params(trainable_params)}")
    print(f"MACs: {format_macs(macs)}")
    print(f"Input size: {input_size}")

    x = torch.randn(input_size).to(device)
    with torch.no_grad():
        output = model(x)
    print(f"Output size: {output.shape}")
    print(f"{'='*50}\n")

    return {
        'model_name': model_name,
        'total_params': total_params,
        'trainable_params': trainable_params,
        'macs': macs,
        'input_size': input_size,
        'output_size': output.shape
    }


def compare_models(models_dict, input_size=(1, 3, 256, 256)):
    results = []
    print(f"\n{'='*60}")
    print(f"Model Comparison (Input size: {input_size})")
    print(f"{'='*60}")

    for name, model in models_dict.items():
        result = calculate_model_overhead(model, name, input_size)
        results.append(result)

    print(f"\n{'='*80}")
    print(f"{'Model Name':<15} {'Parameters':<15} {'MACs':<15}")
    print(f"{'-'*80}")
    for r in results:
        print(f"{r['model_name']:<15} {format_params(r['total_params']):<15} {format_macs(r['macs']):<15}")
    print(f"{'='*80}\n")

    return results


if __name__ == "__main__":
    current_dir = os.path.dirname(os.path.abspath(__file__))
    parent_dir = os.path.dirname(current_dir)
    if parent_dir not in sys.path:
        sys.path.append(parent_dir)
    from models.PGLNet import pglnet_t, pglnet_s, pglnet_b, pglnet_d

    models = {
        'PGLNet-T': pglnet_t(),
        'PGLNet-S': pglnet_s(),
        'PGLNet-B': pglnet_b(),
        'PGLNet-D': pglnet_d(),
    }

    compare_models(models, input_size=(1, 3, 256, 256))

    print("\nTesting different input sizes (PGLNet-T):")
    model = pglnet_t()
    for size in [(1, 3, 256, 256)]:
        calculate_model_overhead(model, f"PGLNet-T {size[-1]}x{size[-1]}", size)
