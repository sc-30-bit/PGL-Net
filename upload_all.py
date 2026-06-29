from huggingface_hub import HfApi

def upload_all_weights():
    api = HfApi()
    
    # 替换为你的 HF 用户名，例如 "sc-30-bit/PGL-Net"
    repo_id = "klay11/PGL-Net" 
    
    print(f"正在确保仓库 {repo_id} 存在...")
    api.create_repo(repo_id=repo_id, repo_type="model", exist_ok=True)
    
    # 这里的路径指向包含 mnn, onnx, pt 等子文件夹的【父文件夹】
    # 请根据你的实际路径修改，如果是当前目录，可以写 "./" 
    # 但建议把它们放在一个单独的文件夹里避免把整个项目代码都传上去
    local_folder_path = r"H:\PGL_Net_Paper_Materials\Deployment-Weights\Deployment-Weights" 
    
    print(f"开始上传 {local_folder_path} 下的所有文件和目录结构...")
    
    # 一键上传整个文件夹
    api.upload_folder(
        folder_path=local_folder_path,
        repo_id=repo_id,
        repo_type="model",
        # 如果你只想传特定格式，可以用 allow_patterns，比如 allow_patterns=["*.onnx", "*.pt", "*.engine"]
        # 默认不填会上传文件夹里的所有内容
    )
    
    print("所有格式的权重已成功同步到 Hugging Face！")

if __name__ == "__main__":
    upload_all_weights()