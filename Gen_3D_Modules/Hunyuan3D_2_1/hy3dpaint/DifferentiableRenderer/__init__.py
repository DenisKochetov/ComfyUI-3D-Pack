# Try to import C++ version first, fallback to Python version
import os
import sys
import subprocess
from pathlib import Path

def try_compile():
    """Attempt to compile the C++ extension"""
    try:
        # Get the directory where this __init__.py is located
        current_dir = Path(__file__).parent.absolute()
        
        # Determine which script to run based on OS
        if os.name == 'nt':  # Windows
            script_name = 'compile_mesh_painter.bat'
        else:  # Linux/Unix/Mac
            script_name = 'compile_mesh_painter.sh'
        
        script_path = current_dir / script_name
        
        if not script_path.exists():
            print(f"[WARNING] Compilation script not found: {script_path}")
            return False
        
        print(f"[INFO] Attempting to compile C++ extension using {script_name}...")
        
        # Run the compilation script from its directory
        result = subprocess.run(
            [str(script_path)] if os.name != 'nt' else ['cmd', '/c', str(script_path)],
            cwd=str(current_dir),
            capture_output=True,
            text=True,
            timeout=300  # 5 minutes timeout
        )
        
        if result.returncode == 0:
            print("[SUCCESS] C++ extension compiled successfully!")
            return True
        else:
            print(f"[ERROR] Compilation failed with return code {result.returncode}")
            if result.stderr:
                print(f"[ERROR] {result.stderr}")
            return False
            
    except Exception as e:
        print(f"[ERROR] Failed to compile C++ extension: {e}")
        return False

# Try importing the compiled module
cpp_imported = False
try:
    from .mesh_inpaint_processor import meshVerticeInpaint
    print("[INFO] Using compiled C++ mesh_inpaint_processor (fast)")
    cpp_imported = True
except ImportError as e:
    # Try to compile and import again
    print("[INFO] C++ extension not found, attempting automatic compilation...")
    if try_compile():
        try:
            from .mesh_inpaint_processor import meshVerticeInpaint
            print("[INFO] Using compiled C++ mesh_inpaint_processor (fast)")
            cpp_imported = True
        except ImportError:
            print("[WARNING] Compilation succeeded but import still failed")

# If C++ version not available, use Python fallback
if not cpp_imported:
    try:
        from .mesh_inpaint_processor_fallback import meshVerticeInpaint
        print("[WARNING] Using Python fallback mesh_inpaint_processor (slower)")
    except ImportError:
        print("[ERROR] Neither C++ nor Python mesh_inpaint_processor found!")
        raise
