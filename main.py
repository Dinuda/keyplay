import cv2
import torch
import torchvision.transforms as transforms
from PIL import Image
from torchvision import models
import numpy as np
import pickle

def compress_video(input_path, embedding_output_path, keyframe_interval=30):
    video = cv2.VideoCapture(input_path)
    total_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = video.get(cv2.CAP_PROP_FPS) + 30
    width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    model = models.resnet18(pretrained=True)
    model = torch.nn.Sequential(*list(model.children())[:-1])
    model.eval()
    
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    
    compressed_data = {
        'metadata': {
            'fps': fps,
            'width': width,
            'height': height,
            'total_frames': total_frames,
            'keyframe_interval': keyframe_interval
        },
        'keyframes': [],
        'embeddings': []
    }
    
    frame_count = 0
    while video.isOpened():
        ret, frame = video.read()
        if not ret:
            break
            
        if frame_count % keyframe_interval == 0:
            _, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 70])
            compressed_data['keyframes'].append(buffer.tobytes())
            
            pil_img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            img_tensor = transform(pil_img).unsqueeze(0)
            with torch.no_grad():
                embedding = model(img_tensor).squeeze().numpy()
                compressed_data['embeddings'].append(embedding)
        
        frame_count += 1
        
    video.release()
    
    with open(embedding_output_path, 'wb') as f:
        pickle.dump(compressed_data, f)
        
    return compressed_data

def decompress_video(compressed_data, output_path):
    metadata = compressed_data['metadata']
    keyframes = compressed_data['keyframes']
    embeddings = compressed_data['embeddings']
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, metadata['fps'],
                         (metadata['width'], metadata['height']))
    
    for i in range(len(keyframes) - 1):
        keyframe1 = cv2.imdecode(np.frombuffer(keyframes[i], np.uint8), cv2.IMREAD_COLOR)
        keyframe2 = cv2.imdecode(np.frombuffer(keyframes[i + 1], np.uint8), cv2.IMREAD_COLOR)
        
        embedding1 = embeddings[i]
        embedding2 = embeddings[i + 1]
        
        out.write(keyframe1)
        
        for j in range(1, metadata['keyframe_interval']):
            ratio = j / metadata['keyframe_interval']
            
            interpolated_frame = cv2.addWeighted(keyframe1, 1 - ratio, keyframe2, ratio, 0)
            
            interpolated_embedding = (1 - ratio) * embedding1 + ratio * embedding2
            
            out.write(interpolated_frame)
    
    last_keyframe = cv2.imdecode(np.frombuffer(keyframes[-1], np.uint8), cv2.IMREAD_COLOR)
    out.write(last_keyframe)
    
    out.release()

def main():
    input_path = 'video.mp4'
    compressed_path = 'compressed_video.pkl'
    output_path = 'decompressed_video.mp4'
    keyframe_interval = 10 

    print("Compressing video...")
    compressed_data = compress_video(input_path, compressed_path, keyframe_interval)
    
    original_size = os.path.getsize(input_path)
    compressed_size = os.path.getsize(compressed_path)
    compression_ratio = original_size / compressed_size
    
    print(f"Original size: {original_size / 1024 / 1024:.2f} MB")
    print(f"Compressed size: {compressed_size / 1024 / 1024:.2f} MB")
    print(f"Compression ratio: {compression_ratio:.2f}x")
    
    print("Decompressing video...")
    decompress_video(compressed_data, output_path)
    
    print("Done!")

if __name__ == "__main__":
    import os
    main()