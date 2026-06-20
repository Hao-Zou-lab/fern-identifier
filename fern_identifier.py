import torch
import clip
from PIL import Image, ImageTk
import tkinter as tk
from tkinter import filedialog, messagebox
import os
import sys
import numpy as np

# ========== CONFIGURATION ==========
WEIGHT_FILE = os.path.join(os.path.dirname(sys.argv[0]), "ensemble_weights.pt")
CLASS_NAMES = [
    "Leptochilus kepingii",
    "Leptochilus digitatus",
    "Other species"
]
CLASS_DISPLAY = {
    "Leptochilus kepingii": "Leptochilus kepingii",
    "Leptochilus digitatus": "Leptochilus digitatus",
    "Other species": "Other species"
}
# ===================================

def resource_path(relative_path):
    if hasattr(sys, '_MEIPASS'):
        return os.path.join(sys._MEIPASS, relative_path)
    return os.path.join(os.path.abspath("."), relative_path)

# Load model
print("Loading model, please wait...")
device = "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)
model = model.float()

class CLIPFineTuner(torch.nn.Module):
    def __init__(self, clip_model, num_classes=3):
        super().__init__()
        self.clip_model = clip_model
        self.fc = torch.nn.Linear(512, num_classes)
    
    def forward(self, images):
        image_features = self.clip_model.encode_image(images)
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        return self.fc(image_features)

# Load weights
print("Loading weights...")
weight_path = WEIGHT_FILE
if not os.path.exists(weight_path):
    messagebox.showerror("Error", f"Weight file not found:\n{weight_path}")
    sys.exit(1)

weights_dict = torch.load(weight_path, map_location=device)

tuners = []
for fold_name, state_dict in weights_dict.items():
    tuner = CLIPFineTuner(model, num_classes=3).to(device)
    tuner.load_state_dict(state_dict)
    tuner.eval()
    tuners.append(tuner)

print(f"Loaded {len(tuners)} models successfully!")

def predict(image_path):
    img = preprocess(Image.open(image_path).convert('RGB')).unsqueeze(0).to(device)
    
    all_probs = []
    with torch.no_grad():
        for tuner in tuners:
            logits = tuner(img)
            probs = torch.softmax(logits, dim=-1)
            all_probs.append(probs.cpu().numpy()[0])
    
    avg_probs = np.mean(all_probs, axis=0)
    pred_idx = np.argmax(avg_probs)
    
    return CLASS_NAMES[pred_idx]

# Create GUI
root = tk.Tk()
root.title("Fern Species Identifier")
root.geometry("500x600")
root.minsize(400, 500)
root.configure(bg='#f0f0f0')

# ================== Scrollable Area ==================
canvas = tk.Canvas(root, bg='#f0f0f0', highlightthickness=0)
scrollbar = tk.Scrollbar(root, orient="vertical", command=canvas.yview)
scrollable_frame = tk.Frame(canvas, bg='#f0f0f0')

scrollable_frame.bind(
    "<Configure>",
    lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
)

canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
canvas.configure(yscrollcommand=scrollbar.set)

canvas.pack(side="left", fill="both", expand=True)
scrollbar.pack(side="right", fill="y")

# ================== Header ==================
title_label = tk.Label(scrollable_frame, text="Fern Species Identifier", font=("Arial", 18, "bold"), bg='#f0f0f0', fg='#2c3e50')
title_label.pack(pady=15)

subtitle_label = tk.Label(scrollable_frame, text="Powered by CLIP (Fine-tuned)", font=("Arial", 10), bg='#f0f0f0', fg='#7f8c8d')
subtitle_label.pack()

# ================== Image Preview Area ==================
preview_frame = tk.Frame(scrollable_frame, bg='#ffffff', bd=2, relief=tk.GROOVE)
preview_frame.pack(pady=15, padx=20, fill=tk.BOTH)

preview_label = tk.Label(preview_frame, text="Image Preview", font=("Arial", 10), bg='#ffffff', fg='#999999')
preview_label.pack(pady=60, padx=80)

# ================== Button Area ==================
btn = tk.Button(scrollable_frame, text="📁 Select Image", 
                font=("Arial", 12), bg="#4CAF50", fg="white", 
                padx=25, pady=6, relief=tk.FLAT)
btn.pack(pady=15)

# ================== Result Area ==================
result_frame = tk.Frame(scrollable_frame, bg='#f0f0f0')
result_frame.pack(pady=15, fill=tk.X)

result_label = tk.Label(result_frame, text="", font=("Arial", 14, "bold"), bg='#f0f0f0', fg='#2196F3')
result_label.pack()

# ================== Footer ==================
info_label = tk.Label(scrollable_frame, text="Recognizable: L. kepingii / L. digitatus", 
                      font=("Arial", 9), bg='#f0f0f0', fg='#95a5a6')
info_label.pack(side="bottom", pady=10)

def show_preview(image_path):
    """Display image preview with size limit"""
    try:
        img = Image.open(image_path)
        # 限制预览大小，保持窗口整洁
        img.thumbnail((350, 350))
        photo = ImageTk.PhotoImage(img)
        preview_label.config(image=photo, text="")
        preview_label.image = photo
    except Exception as e:
        preview_label.config(text=f"Cannot preview: {e}", image='')

def select_image():
    file_path = filedialog.askopenfilename(
        title="Select Image",
        filetypes=[("Image files", "*.jpg *.jpeg *.png *.bmp *.tiff"), ("All files", "*.*")]
    )
    if file_path:
        show_preview(file_path)
        root.title(f"Fern Species Identifier - {os.path.basename(file_path)}")
        # 立即显示"识别中..."提示
        result_label.config(text="Identifying...")
        root.update()
        # 识别
        process_image(file_path)

def process_image(path):
    try:
        result = predict(path)
        display_name = CLASS_DISPLAY.get(result, result)
        result_label.config(text=f"Result: {display_name}", fg='#2196F3')
    except Exception as e:
        messagebox.showerror("Error", f"Failed to recognize: {e}")
        result_label.config(text="Recognition failed", fg='red')

# 绑定按钮命令
btn.config(command=select_image)

root.mainloop()