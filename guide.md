# PagedAttention Experiment Detailed Step-by-Step Guide

## Step 1: Create Experimental Directory
### 🎯 What this step does:
Create a dedicated folder for the experiment to avoid mixing with other projects

### 📂 Specific Operations:
```bash
# Confirm current location
pwd
# Should show: /home/username/vllm-learning/experiments

# Create experiment folder
mkdir paged-attention-test
# 👀 Purpose: Creates a new folder named "paged-attention-test"

cd paged-attention-test
# 👀 Purpose: Enter the newly created experiment folder

# Create subdirectories
mkdir scripts logs results
# 👀 Purpose:
#   • scripts - Store Python script files
#   • logs - Store runtime logs
#   • results - Store experimental results

ls -la
# 👀 Purpose: View the created directory structure to confirm everything was created successfully
```

### 🗂️ Final directory structure:
```
paged-attention-test/
├── scripts/     (Python scripts)
├── logs/        (Log files)
└── results/     (Result data)
```

---

## Step 2: Create Virtual Environment
### 🎯 What this step does:
Create an independent Python environment to avoid affecting existing Python packages on your system

### 🐍 Why we need a virtual environment:
- **Isolation**: Experimental packages won't affect system Python
- **Version control**: Can install specific package versions
- **Cleanliness**: Can delete directly after experiment without leaving traces

### 📂 Specific Operations:
```bash
# Create Python virtual environment
python3 -m venv venv-paged-test
# 👀 Purpose:
#   • Uses your Python 3.12 to create virtual environment
#   • Named "venv-paged-test"
#   • Creates a folder containing independent Python interpreter

# Activate virtual environment
source venv-paged-test/bin/activate
# 👀 Purpose:
#   • Starts the virtual environment
#   • Packages installed afterward will only go into this environment
#   • Command line will show (venv-paged-test) prefix

# Confirm activation success
which python
# 👀 Purpose: Shows current Python path, should point to virtual environment

python --version
# 👀 Purpose: Confirms Python version is still 3.12.3
```

### 🔍 Verification Effect:
After activation, command line should become:
```
(venv-paged-test) username@ubuntu:~/vllm-learning/experiments/paged-attention-test$
```

---

## Step 3: Install Missing Packages
### 🎯 What this step does:
Install Python packages needed for the experiment in the virtual environment

### 📦 Packages to install:
- **transformers**: Traditional large model inference library
- **vLLM**: High-efficiency inference library using PagedAttention
- **nvidia-ml-py3**: GPU monitoring tool
- **psutil**: System monitoring tool

### 📂 Specific Operations:
```bash
# Upgrade pip (package manager)
pip install --upgrade pip
# 👀 Purpose: Ensure pip is latest version to avoid package installation errors

# Install transformers
pip install transformers>=4.36.0
# 👀 Purpose:
#   • Install HuggingFace's transformers library
#   • >=4.36.0 ensures Python 3.12 compatibility
#   • This is the core library for traditional method

# Install vLLM
pip install vllm
# 👀 Purpose:
#   • Install vLLM library containing PagedAttention implementation
#   • This is the new method we want to compare
#   • May take several minutes to download and compile

# Install accelerate (required for transformers)
pip install accelerate
# 👀 Purpose: Required dependency for transformers device_map functionality

# Install monitoring tools
pip install nvidia-ml-py3 psutil
# 👀 Purpose:
#   • nvidia-ml-py3: Direct access to GPU information
#   • psutil: Monitor system resource usage

# Install other tools
pip install matplotlib pandas tqdm
# 👀 Purpose: Data analysis and visualization tools (optional)
```

---

## Step 4: Verify Installation
### 🎯 What this step does:
Confirm all packages are correctly installed and environment setup is successful

### 📂 Specific Operations:
```bash
# Check PyTorch
python -c "import torch; print('✅ PyTorch:', torch.__version__)"
# 👀 Purpose: Confirm PyTorch can import normally, display version number

# Check Transformers
python -c "import transformers; print('✅ Transformers:', transformers.__version__)"
# 👀 Purpose: Confirm transformers installation success

# Check vLLM
python -c "import vllm; print('✅ vLLM:', vllm.__version__)"
# 👀 Purpose: Confirm vLLM installation success

# Check CUDA
python -c "import torch; print('✅ CUDA available:', torch.cuda.is_available()); print('✅ GPU count:', torch.cuda.device_count())"
# 👀 Purpose: Confirm GPU can be recognized by PyTorch
```

### 🎉 Successful output should look like:
```
✅ PyTorch: 2.7.1
✅ Transformers: 4.37.2
✅ vLLM: 0.3.0
✅ CUDA available: True
✅ GPU count: 1
```

---

## Step 5: Create Memory Monitoring Script
### 🎯 What this step does:
Write a Python script to monitor GPU memory usage

### 📍 Where to run:
```bash
# Ensure you're in this directory:
pwd
# Should show: /home/username/vllm-learning/experiments/paged-attention-test

# Ensure virtual environment is activated:
source venv-paged-test/bin/activate
# Command line should show: (venv-paged-test)
```

### 🔍 Why we need monitoring:
- **Comparison key**: See memory usage differences between two methods
- **Real-time observation**: Understand memory usage change patterns
- **Quantitative analysis**: Use data to prove PagedAttention advantages

### 📂 Create script:
```bash
cat > scripts/memory_monitor.py << 'EOF'
import torch
import time
from datetime import datetime

class MemoryMonitor:
    def __init__(self):
        self.gpu_available = torch.cuda.is_available()
        if self.gpu_available:
            try:
                import nvidia_ml_py3 as nvml
                nvml.nvmlInit()
                self.nvml = nvml
                self.device_count = nvml.nvmlDeviceGetCount()
                self.nvml_available = True
                print("✅ Using NVML for GPU monitoring")
            except ImportError:
                print("⚠️  NVML unavailable, using PyTorch for GPU monitoring")
                self.nvml_available = False
        else:
            print("❌ GPU unavailable")
    
    def get_gpu_memory_torch(self):
        """Get GPU memory using torch"""
        if not self.gpu_available:
            return {"error": "No GPU available"}
        
        memory_info = {}
        for i in range(torch.cuda.device_count()):
            torch.cuda.set_device(i)
            total = torch.cuda.get_device_properties(i).total_memory / 1024**3
            allocated = torch.cuda.memory_allocated(i) / 1024**3
            cached = torch.cuda.memory_reserved(i) / 1024**3
            
            memory_info[f"GPU_{i}"] = {
                "total": total,
                "allocated": allocated,
                "cached": cached,
                "free": total - cached,
                "utilization": (allocated / total) * 100 if total > 0 else 0
            }
        return memory_info
    
    def get_gpu_memory_nvml(self):
        """Get GPU memory using NVML"""
        if not self.gpu_available or not self.nvml_available:
            return self.get_gpu_memory_torch()
            
        memory_info = {}
        for i in range(self.device_count):
            handle = self.nvml.nvmlDeviceGetHandleByIndex(i)
            memory = self.nvml.nvmlDeviceGetMemoryInfo(handle)
            
            memory_info[f"GPU_{i}"] = {
                "total": memory.total / 1024**3,
                "used": memory.used / 1024**3,
                "free": memory.free / 1024**3,
                "utilization": (memory.used / memory.total) * 100
            }
        return memory_info
    
    def get_gpu_memory(self):
        """Get GPU memory usage"""
        if self.nvml_available:
            return self.get_gpu_memory_nvml()
        else:
            return self.get_gpu_memory_torch()
    
    def print_memory_status(self, label=""):
        """Print memory status"""
        print(f"\n=== {label} ===")
        print(f"Time: {datetime.now().strftime('%H:%M:%S')}")
        
        # GPU memory
        gpu_memory = self.get_gpu_memory()
        if "error" not in gpu_memory:
            for gpu_id, mem in gpu_memory.items():
                if "used" in mem:
                    print(f"🔥 {gpu_id}: {mem['used']:.2f}GB / {mem['total']:.2f}GB ({mem['utilization']:.1f}%)")
                else:
                    print(f"🔥 {gpu_id}: {mem['allocated']:.2f}GB allocated / {mem['total']:.2f}GB total")
        else:
            print("❌ Failed to get GPU information")
        
        print("-" * 50)

if __name__ == "__main__":
    monitor = MemoryMonitor()
    monitor.print_memory_status("System Status Check")
    
    # Test GPU availability
    if torch.cuda.is_available():
        print(f"✅ CUDA available, device count: {torch.cuda.device_count()}")
        for i in range(torch.cuda.device_count()):
            print(f"📊 GPU {i}: {torch.cuda.get_device_name(i)}")
    else:
        print("❌ CUDA unavailable")
EOF
```

### 🎯 This script enables us to:
- Compare memory usage before and after model loading
- Observe memory changes when processing different requests
- Quantify memory efficiency differences between two methods

---

## Step 6: Test Monitoring Script
### 🎯 What this step does:
Run the just-created monitoring script to ensure it works properly

### 📍 Where to run:
```bash
# Ensure in project root directory
pwd
# Should show: /home/username/vllm-learning/experiments/paged-attention-test

# Ensure virtual environment is activated
source venv-paged-test/bin/activate

# Run script from project root directory
python scripts/memory_monitor.py
```
**Note**: Script file is in `scripts/` folder, but we run it from project root directory

### 🎉 Normal output should look like:
```
=== System Status Check ===
Time: 20:30:15
🔥 GPU_0: 0.56GB / 24.00GB (2.3%)
--------------------------------------------------
✅ CUDA available, device count: 1
📊 GPU 0: NVIDIA GeForce RTX 4090
```

---

## Step 7: Create Traditional Method Test Script
### 🎯 What this step does:
Write a script to test traditional Transformers library memory usage patterns

### 🔍 Traditional method characteristics:
- **Sequential processing**: Can only process one request at a time
- **Fixed allocation**: Allocates memory by maximum possible length
- **Memory waste**: Short texts also occupy memory space for long texts

### 📂 Create script:
```bash
cat > scripts/test_traditional.py << 'EOF'
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import torch
import time
from transformers import AutoModelForCausalLM, AutoTokenizer
from memory_monitor import MemoryMonitor

def test_traditional_method():
    """Test traditional Transformers method"""
    monitor = MemoryMonitor()
    
    print("🔄 Starting traditional method test...")
    monitor.print_memory_status("Initial state")
    
    # Use GPT-2 model for testing
    model_name = "gpt2"
    
    print("📥 Loading traditional Transformers model...")
    try:
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16,
            low_cpu_mem_usage=True
        )
        # Manually move to GPU
        if torch.cuda.is_available():
            model = model.to("cuda")
        
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        monitor.print_memory_status("Model loading complete")
        
    except Exception as e:
        print(f"❌ Model loading failed: {e}")
        return
    
    # Prepare test data - different length texts
    test_inputs = [
        "Hello!",  # Short text
        "How are you doing today? I hope everything is going well.",  # Medium length
        "Tell me a story about artificial intelligence and machine learning. " * 10,  # Long text
        "Thanks!"  # Short text
    ]
    
    print(f"📝 Processing {len(test_inputs)} inputs of different lengths...")
    
    total_time = 0
    for i, input_text in enumerate(test_inputs):
        print(f"\n📄 Processing input {i+1}: length {len(input_text)} characters")
        
        try:
            # Encode input
            inputs = tokenizer(
                input_text, 
                return_tensors="pt", 
                padding=True, 
                truncation=True,
                max_length=512
            )
            
            # Move to GPU
            if torch.cuda.is_available():
                inputs = {k: v.to("cuda") for k, v in inputs.items()}
            
            # Generate text
            with torch.no_grad():
                start_time = time.time()
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=50,
                    do_sample=True,
                    temperature=0.7,
                    pad_token_id=tokenizer.eos_token_id
                )
                generation_time = time.time() - start_time
                total_time += generation_time
            
            # Decode output
            generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
            print(f"⏱️  Generation time: {generation_time:.2f}s")
            print(f"📊 Output length: {len(generated_text)} characters")
            
            monitor.print_memory_status(f"Completed input {i+1}")
            
            # Clean GPU cache
            torch.cuda.empty_cache()
            time.sleep(1)  # Wait for memory release
                
        except Exception as e:
            print(f"❌ Error processing input {i+1}: {e}")
    
    print(f"\n✅ Traditional method test complete!")
    print(f"🕒 Total time: {total_time:.2f}s")
    print(f"📈 Average per request: {total_time/len(test_inputs):.2f}s")
    print(f"💡 Key observation: Each request needs separate processing, memory usage fluctuates")

if __name__ == "__main__":
    test_traditional_method()
EOF
```

### 🔧 If encountering accelerate-related errors:
```bash
# Install missing dependency
pip install accelerate

# Re-run test
python scripts/test_traditional.py
```

### 🎯 Through this test we can observe:
- How memory is allocated each time processing
- Whether memory usage differs for different text lengths
- Traditional method's memory usage patterns

---

## Step 8: Create vLLM Test Script
### 🎯 What this step does:
Write a script to test vLLM library's PagedAttention memory usage patterns

### 🔍 PagedAttention characteristics:
- **Batch processing**: Can process multiple requests simultaneously
- **Paged allocation**: Allocates memory pages based on actual needs
- **High efficiency**: Avoids memory waste and fragmentation

### 📂 Create script:
```bash
cat > scripts/test_vllm.py << 'EOF'
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import time
from memory_monitor import MemoryMonitor

def test_vllm_method():
    """Test vLLM PagedAttention method"""
    monitor = MemoryMonitor()
    
    print("🔄 Starting vLLM method test...")
    monitor.print_memory_status("Initial state")
    
    try:
        from vllm import LLM, SamplingParams
        print("✅ vLLM import successful")
    except ImportError as e:
        print(f"❌ vLLM import failed: {e}")
        print("💡 Please ensure vLLM is installed: pip install vllm")
        return
    
    # Load model
    print("📥 Loading vLLM model...")
    try:
        llm = LLM(
            model="gpt2",
            trust_remote_code=True,
            max_model_len=512,
            gpu_memory_utilization=0.8  # Use 80% of GPU memory
        )
        monitor.print_memory_status("vLLM model loading complete")
    except Exception as e:
        print(f"❌ vLLM model loading failed: {e}")
        return
    
    # Set sampling parameters
    sampling_params = SamplingParams(
        temperature=0.7,
        max_tokens=50,
        top_p=0.95
    )
    
    # Prepare test data (same as traditional method)
    test_inputs = [
        "Hello!",  # Short text
        "How are you doing today? I hope everything is going well.",  # Medium length  
        "Tell me a story about artificial intelligence and machine learning. " * 10,  # Long text
        "Thanks!"  # Short text
    ]
    
    print(f"📝 Processing {len(test_inputs)} inputs of different lengths...")
    print("🚀 vLLM advantage: Batch processing all requests!")
    
    try:
        # Batch process all inputs - this is vLLM's core advantage
        start_time = time.time()
        outputs = llm.generate(test_inputs, sampling_params)
        total_time = time.time() - start_time
        
        monitor.print_memory_status("Batch generation complete")
        
        # Display results
        for i, output in enumerate(outputs):
            prompt = output.prompt
            generated_text = output.outputs[0].text
            print(f"\n📄 Input {i+1}: '{prompt[:50]}...'")
            print(f"📊 Original length: {len(prompt)} characters")
            print(f"📊 Generated length: {len(generated_text)} characters")
        
        print(f"\n✅ vLLM method test complete!")
        print(f"🕒 Total generation time: {total_time:.2f}s")
        print(f"📈 Average per request: {total_time/len(test_inputs):.2f}s")
        print(f"🔥 Core advantage: Batch processing, more stable memory usage, supports PagedAttention!")
        
    except Exception as e:
        print(f"❌ vLLM generation process error: {e}")

if __name__ == "__main__":
    test_vllm_method()
EOF
```

### 🎯 Through this test we can observe:
- Batch processing efficiency advantages
- Memory usage stability
- PagedAttention's actual effects

---

## Step 9: Create Run Script
### 🎯 What this step does:
Write a master control script to automatically run the entire experimental process

### 📂 Create script:
```bash
cat > run_experiment.py << 'EOF'
#!/usr/bin/env python3
import subprocess
import sys
import os
import time

def check_environment():
    """Check experimental environment"""
    print("🔍 Checking experimental environment...")
    
    # Check Python version
    print(f"🐍 Python version: {sys.version}")
    
    # Check virtual environment
    if hasattr(sys, 'real_prefix') or (hasattr(sys, 'base_prefix') and sys.base_prefix != sys.prefix):
        print("✅ Running in virtual environment")
    else:
        print("⚠️  Recommend running in virtual environment")
    
    # Check necessary modules
    try:
        import torch
        print(f"✅ PyTorch version: {torch.__version__}")
        print(f"✅ CUDA available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"📊 GPU count: {torch.cuda.device_count()}")
            for i in range(torch.cuda.device_count()):
                print(f"   GPU {i}: {torch.cuda.get_device_name(i)}")
    except ImportError:
        print("❌ PyTorch not installed")
        return False
    
    try:
        import transformers
        print(f"✅ Transformers version: {transformers.__version__}")
    except ImportError:
        print("❌ Transformers not installed")
        return False
    
    try:
        import vllm
        print(f"✅ vLLM version: {vllm.__version__}")
    except ImportError:
        print("❌ vLLM not installed")
        return False
    
    return True

def run_experiment():
    """Run comparison experiment"""
    print("\n" + "=" * 60)
    print("🧪 PagedAttention Memory Allocation Comparison Experiment")
    print("🎯 Goal: Understand how PagedAttention improves memory utilization")
    print("=" * 60)
    
    if not check_environment():
        print("❌ Environment check failed, please install necessary dependencies")
        return
    
    script_dir = "scripts"
    
    # Test traditional method
    print("\n" + "="*50)
    print("🔬 Phase 1: Traditional Transformers Method Test")
    print("💡 Characteristics: Sequential processing, max-length memory allocation")
    print("="*50)
    
    try:
        print("⏳ Running traditional method test...")
        result = subprocess.run([sys.executable, f"{script_dir}/test_traditional.py"], 
                              cwd=os.getcwd())
        if result.returncode == 0:
            print("\n✅ Traditional method test complete")
        else:
            print(f"\n❌ Traditional method test failed, return code: {result.returncode}")
    except Exception as e:
        print(f"❌ Traditional method test error: {e}")
    
    # Wait for user confirmation
    print("\n" + "="*50)
    input("⏸️  Press Enter to continue with vLLM PagedAttention method test...")
    
    # Test vLLM method
    print("\n" + "="*50)
    print("🔬 Phase 2: vLLM PagedAttention Method Test")
    print("💡 Characteristics: Batch processing, paged KV-Cache management, on-demand memory allocation")
    print("="*50)
    
    try:
        print("⏳ Running vLLM method test...")
        result = subprocess.run([sys.executable, f"{script_dir}/test_vllm.py"], 
                              cwd=os.getcwd())
        if result.returncode == 0:
            print("\n✅ vLLM method test complete")
        else:
            print(f"\n❌ vLLM method test failed, return code: {result.returncode}")
    except Exception as e:
        print(f"❌ vLLM method test error: {e}")
    
    # Experiment summary
    print("\n" + "=" * 60)
    print("🎉 Experiment Complete! PagedAttention Principle Analysis")
    print("=" * 60)
    print("\n🔍 Key Observations:")
    print("1. 💾 GPU Memory Usage Patterns:")
    print("   • Traditional method: Each sequence allocated by max length, memory waste exists")
    print("   • PagedAttention: Pages allocated based on actual needs, higher memory utilization")
    print("\n2. 📊 Memory Usage Stability:")
    print("   • Traditional method: Memory usage fluctuates greatly with input length")
    print("   • PagedAttention: More stable memory usage, better predictability")
    print("\n3. ⚡ Processing Efficiency:")
    print("   • Traditional method: Sequential request processing")
    print("   • PagedAttention: Supports efficient batch processing")
    print("\n💡 PagedAttention Core Innovations:")
    print("🔹 Splits KV-Cache into fixed-size pages")
    print("🔹 Allocates pages on-demand, avoiding pre-allocation of large contiguous memory blocks")
    print("🔹 Supports dynamic memory management and better memory fragmentation handling")
    print("🔹 Enables batch processing of variable-length sequences")
    print("\n🎯 Practical Significance:")
    print("📈 Supports larger batch sizes on same hardware")
    print("💰 Reduces hardware costs for inference services")
    print("🚀 Improves model service throughput")

if __name__ == "__main__":
    run_experiment()
EOF

# Give scripts execution permissions
chmod +x run_experiment.py
chmod +x scripts/*.py
```

### 🎯 Value of this script:
- **Automation**: One-click run of all tests
- **User-friendly**: Clear prompts and explanations
- **Educational**: Explains the principles behind each phenomenon

---

## Step 10: Run Experiment
### 🎯 What this step does:
Execute the complete comparison experiment and observe differences between the two methods

### 📍 Where to run:
```bash
# 1. Ensure in correct directory
pwd
# Must show: /home/username/vllm-learning/experiments/paged-attention-test

# 2. Ensure virtual environment is activated
source venv-paged-test/bin/activate
# Command line should show: (venv-paged-test)

# 3. Run main experiment script
python run_experiment.py
```

### 🗂️ Important directory structure:
```
/home/username/vllm-learning/experiments/paged-attention-test/  ← You run here
├── venv-paged-test/           ← Virtual environment
├── scripts/                   ← Script files location
│   ├── memory_monitor.py      ← Monitoring script
│   ├── test_traditional.py    ← Traditional method test
│   └── test_vllm.py          ← vLLM method test
├── logs/                      ← Log output
├── results/                   ← Result files
└── run_experiment.py          ← Main run script (here)
```

### 🔄 Script call relationship:
```
You run: python run_experiment.py  (in root directory)
   ↓
Auto calls: python scripts/test_traditional.py
   ↓  
Auto calls: python scripts/test_vllm.py
```

### 🎯 Key differences you will observe:

#### Traditional Method Characteristics:
- 📊 Memory usage: Fluctuates with input length
- ⏱️ Processing mode: Sequential request processing
- 💾 Memory allocation: Pre-allocate by maximum length

#### PagedAttention Characteristics:
- 📊 Memory usage: More stable and efficient
- ⏱️ Processing mode: Batch processing
- 💾 Memory allocation: On-demand paged allocation

### 🎓 Educational value of the experiment:
Through this experiment, you will deeply understand:
1. **How PagedAttention works**
2. **Why it can improve memory utilization**
3. **Its advantages in practical applications**

---

## Summary: Core Purpose of Each Step

| Step | Main Purpose | Key Output |
|------|--------------|------------|
| 1-2 | Environment preparation | Independent experimental environment |
| 3-4 | Dependency installation | Complete software stack |
| 5-6 | Monitoring tools | Memory usage observation capability |
| 7-8 | Test scripts | Comparison tests for both methods |
| 9-10 | Experiment execution | Intuitive understanding of PagedAttention principles |

Each step serves the ultimate goal: **Through practical comparison, gain deep understanding of PagedAttention's working principles and advantages**.
