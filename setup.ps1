# setup.ps1 - BEHAVIOR-1K Installation Script (Windows)
param(
    [switch]$Help,
    [switch]$NewEnv,
    [switch]$OmniGibson,
    [switch]$BDDL,
    [switch]$Teleop,
    [switch]$Dataset,
    [switch]$Primitives,
    [switch]$Eval,
    [switch]$Dev,
    [string]$CudaVersion = "12.4",
    [switch]$AcceptCondaTos,
    [switch]$AcceptNvidiaEula,
    [switch]$AcceptDatasetTos,
    [switch]$ConfirmNoConda
)

# Set error action preference to stop on errors
$ErrorActionPreference = "Stop"

# Show help if no parameters or help requested
if ($Help -or $PSBoundParameters.Count -eq 0) {
    Write-Host @"
BEHAVIOR-1K Installation Script (Windows)
Usage: .\setup.ps1 [OPTIONS]

Options:
  -Help                   Display this help message
  -NewEnv                 Create a new conda environment 'behavior'
  -OmniGibson             Install OmniGibson (core physics simulator)
  -BDDL                   Install BDDL (Behavior Domain Definition Language)
  -Teleop                 Install JoyLo (teleoperation interface)
  -Dataset                Download BEHAVIOR datasets (requires -OmniGibson)
  -Primitives             Install OmniGibson with primitives support
  -Eval                   Install evaluation dependencies
  -Dev                    Install development dependencies
  -CudaVersion VERSION    Specify CUDA version (default: 12.4)
  -AcceptCondaTos         Automatically accept Conda Terms of Service
  -AcceptNvidiaEula       Automatically accept NVIDIA Isaac Sim EULA
  -AcceptDatasetTos       Automatically accept BEHAVIOR Dataset Terms
  -ConfirmNoConda         Skip confirmation prompt when not in a conda environment

Example: .\setup.ps1 -NewEnv -OmniGibson -BDDL -Teleop -Dataset
Example (non-interactive): .\setup.ps1 -NewEnv -OmniGibson -Dataset -AcceptCondaTos -AcceptNvidiaEula -AcceptDatasetTos
"@
    exit 0
}

# Validate dependencies
if ($OmniGibson -and -not $BDDL) {
    Write-Error "ERROR: -OmniGibson requires -BDDL"
    exit 1
}

if ($Dataset -and -not $OmniGibson) {
    Write-Error "ERROR: -Dataset requires -OmniGibson"
    exit 1
}

if ($Primitives -and -not $OmniGibson) {
    Write-Error "ERROR: -Primitives requires -OmniGibson"
    exit 1
}

if ($Eval -and -not $OmniGibson) {
    Write-Error "ERROR: -Eval requires -OmniGibson"
}

if ($NewEnv -and $ConfirmNoConda) {
    Write-Error "ERROR: -NewEnv and -ConfirmNoConda are mutually exclusive"
    exit 1
}

$WorkDir = Get-Location

# Check conda environment condition early (unless creating new environment)
if (-not $NewEnv) {
    if (-not $env:CONDA_PREFIX) {
        if (-not $ConfirmNoConda) {
            Write-Host ""
            Write-Host "WARNING: You are not in a conda environment."
            Write-Host "Currently using Python from: $(Get-Command python | Select-Object -ExpandProperty Source)"
            Write-Host ""
            Write-Host "Continue? [y/n] (or rerun with -ConfirmNoConda to skip this prompt)"
            $response = Read-Host
            if ($response -notmatch '^[Yy]$') {
                Write-Host "Installation cancelled."
                exit 1
            }
        }
        Write-Host "Proceeding without conda environment..."
    }
}

# Function to prompt for terms acceptance
function Prompt-ForTerms {
    Write-Host ""
    Write-Host "=== TERMS OF SERVICE AND LICENSING AGREEMENTS ==="
    Write-Host ""
    
    # Check what terms need to be accepted
    $NeedsCondaTos = $false
    $NeedsNvidiaEula = $false
    $NeedsDatasetTos = $false
    
    if ($NewEnv -and -not $AcceptCondaTos) {
        $script:NeedsCondaTos = $true
    }
    
    if ($OmniGibson -and -not $AcceptNvidiaEula) {
        $script:NeedsNvidiaEula = $true
    }
    
    if ($Dataset -and -not $AcceptDatasetTos) {
        $script:NeedsDatasetTos = $true
    }
    
    # If nothing needs acceptance, return early
    if (-not $script:NeedsCondaTos -and -not $script:NeedsNvidiaEula -and -not $script:NeedsDatasetTos) {
        return
    }
    
    Write-Host "This installation requires acceptance of the following terms:"
    Write-Host ""
    
    if ($script:NeedsCondaTos) {
        Write-Host @"
1. CONDA TERMS OF SERVICE
   - Required for creating conda environment
   - By accepting, you agree to Anaconda's Terms of Service
   - See: https://legal.anaconda.com/policies/en/

"@
    }
    
    if ($script:NeedsNvidiaEula) {
        Write-Host @"
2. NVIDIA ISAAC SIM EULA
   - Required for OmniGibson installation
   - By accepting, you agree to NVIDIA Isaac Sim End User License Agreement
   - See: https://www.nvidia.com/en-us/agreements/enterprise-software/nvidia-software-license-agreement

"@
    }
    
    if ($script:NeedsDatasetTos) {
        Write-Host @"
3. BEHAVIOR DATA BUNDLE END USER LICENSE AGREEMENT
    Last revision: December 8, 2022
    This License Agreement is for the BEHAVIOR Data Bundle ("Data"). It works with OmniGibson ("Software") which is a software stack licensed under the MIT License, provided in this repository: https://github.com/StanfordVL/OmniGibson. 
    The license agreements for OmniGibson and the Data are independent. This BEHAVIOR Data Bundle contains artwork and images ("Third Party Content") from third parties with restrictions on redistribution. 
    It requires measures to protect the Third Party Content which we have taken such as encryption and the inclusion of restrictions on any reverse engineering and use. 
    Recipient is granted the right to use the Data under the following terms and conditions of this License Agreement ("Agreement"):
        1. Use of the Data is permitted after responding "Yes" to this agreement. A decryption key will be installed automatically.
        2. Data may only be used for non-commercial academic research. You may not use a Data for any other purpose.
        3. The Data has been encrypted. You are strictly prohibited from extracting any Data from OmniGibson or reverse engineering.
        4. You may only use the Data within OmniGibson.
        5. You may not redistribute the key or any other Data or elements in whole or part.
        6. THE DATA AND SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. 
            IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE DATA OR SOFTWARE OR THE USE OR OTHER DEALINGS IN THE DATA OR SOFTWARE.

"@
    }
    
    Write-Host "Do you accept ALL of the above terms? (y/N)"
    $response = Read-Host
    
    if ($response -notmatch '^[Yy]$') {
        Write-Host "Terms not accepted. Installation cancelled."
        Write-Host "You can bypass these prompts by using -AcceptCondaTos, -AcceptNvidiaEula, and -AcceptDatasetTos flags."
        exit 1
    }
    
    # Set acceptance flags
    if ($script:NeedsCondaTos) { $script:AcceptCondaTos = $true }
    if ($script:NeedsNvidiaEula) { $script:AcceptNvidiaEula = $true }
    if ($script:NeedsDatasetTos) { $script:AcceptDatasetTos = $true }
    
    Write-Host ""
    Write-Host "✓ All terms accepted. Proceeding with installation..."
    Write-Host ""
}

# Prompt for terms acceptance at the beginning
Prompt-ForTerms

# Helper function to check if conda environment exists
function Test-CondaEnvironment {
    param([string]$EnvName)
    $envList = conda env list 2>$null | Out-String
    return $envList -match "^$EnvName\s"
}

# Helper function to activate conda environment
function Invoke-CondaActivate {
    param([string]$EnvName)
    
    # Get conda base path
    $condaInfo = conda info --base 2>$null
    if (-not $condaInfo) {
        Write-Error "ERROR: Failed to get conda info"
        exit 1
    }
    
    # Source conda initialization
    $condaHook = "$condaInfo\shell\condabin\conda-hook.ps1"
    if (Test-Path $condaHook) {
        & $condaHook
    }
    
    conda activate $EnvName
    
    if ($env:CONDA_DEFAULT_ENV -ne $EnvName) {
        Write-Error "ERROR: Failed to activate environment '$EnvName'"
        exit 1
    }
}

# Create conda environment
if ($NewEnv) {
    Write-Host "Creating conda environment 'behavior'..."
    
    # Check if conda is available
    try {
        conda --version | Out-Null
    }
    catch {
        Write-Error "ERROR: Conda not found"
        exit 1
    }
    
    # Set auto-accept environment variable if user agreed to TOS
    if ($AcceptCondaTos) {
        $env:CONDA_PLUGINS_AUTO_ACCEPT_TOS = "yes"
        Write-Host "✓ Conda TOS auto-acceptance enabled"
    }
    
    # Check if environment already exists and exit with instructions
    if (Test-CondaEnvironment "behavior") {
        Write-Host ""
        Write-Host "ERROR: Conda environment 'behavior' already exists!"
        Write-Host ""
        Write-Host "Please remove or rename the existing environment and re-run this script."
        Write-Host ""
        exit 1
    }
    
    # Create environment with only Python 3.10
    conda create -n behavior python=3.10 -c conda-forge -y
    
    # Activate environment
    Invoke-CondaActivate "behavior"
    
    # Install PyTorch via pip with CUDA support
    Write-Host "Installing PyTorch with CUDA $CudaVersion support..."
    
    # Determine the CUDA version string for pip URL (e.g., cu126, cu118, etc.)
    $CudaVerShort = $CudaVersion -replace '\.', ''  # Convert 12.4 to 124
    
    # Install numpy and setuptools via pip
    Write-Host "Installing numpy and setuptools..."
    pip install "numpy<2" "setuptools<=79"
    
    pip install torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0 --index-url "https://download.pytorch.org/whl/cu$CudaVerShort"
    
    Write-Host "✓ PyTorch installation completed"
}

# Setup Isaac Sim conda environment helper functions
function Set-IsaacSimCondaEnv {
    param(
        [string]$IsaacSimPath,
        [string]$CondaPrefix
    )
    
    # Create directories
    $activateDir = Join-Path $CondaPrefix "etc\conda\activate.d"
    $deactivateDir = Join-Path $CondaPrefix "etc\conda\deactivate.d"
    
    New-Item -ItemType Directory -Path $activateDir -Force | Out-Null
    New-Item -ItemType Directory -Path $deactivateDir -Force | Out-Null
    
    # Create empty files
    foreach ($dir in @($activateDir, $deactivateDir)) {
        foreach ($file in @("env_vars.bat", "env_vars.ps1")) {
            $filePath = Join-Path $dir $file
            if (-not (Test-Path $filePath)) {
                New-Item -ItemType File -Path $filePath -Force | Out-Null
            }
        }
    }
    
    # Setup CMD activation script
    $cmdActFile = Join-Path $activateDir "env_vars.bat"
    @"
@echo off
set PYTHONPATH_OLD=%PYTHONPATH%
set PYTHONPATH=%PYTHONPATH%;$IsaacSimPath\site
set CARB_APP_PATH=$IsaacSimPath\kit
set EXP_PATH=$IsaacSimPath\apps
set ISAAC_PATH=$IsaacSimPath
"@ | Out-File -FilePath $cmdActFile -Encoding ascii
    
    # Setup CMD deactivation script
    $cmdDeactFile = Join-Path $deactivateDir "env_vars.bat"
    @"
@echo off
set PYTHONPATH=%PYTHONPATH_OLD%
set PYTHONPATH_OLD=""
"@ | Out-File -FilePath $cmdDeactFile -Encoding ascii
    
    # Setup PowerShell activation script
    $psActFile = Join-Path $activateDir "env_vars.ps1"
    @"
`$env:PYTHONPATH_OLD="`$env:PYTHONPATH"
`$env:PYTHONPATH="`$env:PYTHONPATH;$IsaacSimPath\site"
`$env:CARB_APP_PATH="$IsaacSimPath\kit"
`$env:EXP_PATH="$IsaacSimPath\apps"
`$env:ISAAC_PATH="$IsaacSimPath"
"@ | Out-File -FilePath $psActFile -Encoding utf8
    
    # Setup PowerShell deactivation script
    $psDeactFile = Join-Path $deactivateDir "env_vars.ps1"
    @"
`$env:PYTHONPATH="`$env:PYTHONPATH_OLD"
`$env:PYTHONPATH_OLD=`$null
"@ | Out-File -FilePath $psDeactFile -Encoding utf8
}

# Find Isaac Sim installation
function Find-IsaacSimPath {
    $basePath = Join-Path $env:USERPROFILE "AppData\Local\ov\pkg"
    
    if (-not (Test-Path $basePath)) {
        return $null
    }
    
    $isaacDirs = Get-ChildItem -Path $basePath -Directory -Name "isaac*" | Sort-Object
    if ($isaacDirs.Count -eq 0) {
        return $null
    }
    
    return Join-Path $basePath $isaacDirs[-1]
}

# Install BDDL
if ($BDDL) {
    Write-Host "Installing BDDL..."
    
    if (-not (Test-Path "bddl")) {
        Write-Error "ERROR: bddl directory not found"
        exit 1
    }
    
    pip install -e "$WorkDir\bddl"
}

# Install OmniGibson with Isaac Sim
if ($OmniGibson) {
    Write-Host "Installing OmniGibson..."
    
    if (-not (Test-Path "OmniGibson")) {
        Write-Error "ERROR: OmniGibson directory not found"
        exit 1
    }
    
    # Check Python version
    $pythonVersion = python -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')"
    if ($pythonVersion -ne "3.10") {
        Write-Error "ERROR: Python 3.10 required, found $pythonVersion"
        exit 1
    }
    
    # Check for conflicting environment variables
    if ($env:EXP_PATH -or $env:CARB_APP_PATH -or $env:ISAAC_PATH) {
        Write-Error @"
ERROR: Found existing Isaac Sim environment variables.
Please unset EXP_PATH, CARB_APP_PATH, and ISAAC_PATH and restart.
"@
        exit 1
    }
    
    # Build extras string
$extrasList = @()

if ($Dev) { $extrasList += "dev" }
if ($Primitives) { $extrasList += "primitives" }
if ($Eval) { $extrasList += "eval" }

if ($extrasList.Count -gt 0) {
    $extras = "[" + ($extrasList -join ",") + "]"
} else {
    $extras = ""
}
    
    pip install -e "$WorkDir\OmniGibson$extras"
    
    # Install pre-commit for dev setup
    if ($Dev) {
        Write-Host "Setting up pre-commit..."
        conda install -c conda-forge pre-commit -y
        Push-Location "$WorkDir\OmniGibson"
        pre-commit install
        Pop-Location
    }
    
    # Isaac Sim installation via pip
    if ($AcceptNvidiaEula) {
        $env:OMNI_KIT_ACCEPT_EULA = "YES"
    }
    else {
        Write-Error "ERROR: NVIDIA EULA not accepted. Cannot install Isaac Sim."
        exit 1
    }
    
    # Check if already installed
    $isaacInstalled = $false
    try {
        python -c "import isaacsim" 2>$null
        $isaacInstalled = $true
        Write-Host "Isaac Sim already installed, skipping..."
    }
    catch {
        Write-Host "Installing Isaac Sim via pip..."
    }
    
if (-not $isaacInstalled) {
        # Isaac Sim packages to install
        $packages = @(
            "omniverse_kit-106.5.0.162521", "isaacsim_kernel-4.5.0.0", "isaacsim_app-4.5.0.0",
            "isaacsim_core-4.5.0.0", "isaacsim_gui-4.5.0.0", "isaacsim_utils-4.5.0.0",
            "isaacsim_storage-4.5.0.0", "isaacsim_asset-4.5.0.0", "isaacsim_sensor-4.5.0.0",
            "isaacsim_robot_motion-4.5.0.0", "isaacsim_robot-4.5.0.0", "isaacsim_benchmark-4.5.0.0",
            "isaacsim_code_editor-4.5.0.0", "isaacsim_ros1-4.5.0.0", "isaacsim_cortex-4.5.0.0",
            "isaacsim_example-4.5.0.0", "isaacsim_replicator-4.5.0.0", "isaacsim_rl-4.5.0.0",
            "isaacsim_robot_setup-4.5.0.0", "isaacsim_ros2-4.5.0.0", "isaacsim_template-4.5.0.0",
            "isaacsim_test-4.5.0.0", "isaacsim-4.5.0.0", "isaacsim_extscache_physics-4.5.0.0",
            "isaacsim_extscache_kit-4.5.0.0", "isaacsim_extscache_kit_sdk-4.5.0.0"
        )
        
        $tempDir = New-TemporaryFile | ForEach-Object { Remove-Item $_; New-Item -ItemType Directory -Path $_ }
        $wheelFiles = @()
        
        try {
            foreach ($pkg in $packages) {
                $pkgParts = $pkg -split "-"
                $pkgName = ($pkgParts[0..$($pkgParts.Length-2)] -join "-").Replace("_", "-")
                $filename = "$pkg-cp310-none-win_amd64.whl"
                $url = "https://pypi.nvidia.com/$pkgName/$filename"
                $filepath = Join-Path $tempDir $filename
                
                Write-Host "Downloading $pkg..."
                try {
                    Invoke-WebRequest -Uri $url -OutFile $filepath -UseBasicParsing
                    $wheelFiles += $filepath
                }
                catch {
                    Write-Error "ERROR: Failed to download $pkg"
                    exit 1
                }
            }
            
            Write-Host "Installing Isaac Sim packages..."
            pip install $wheelFiles
            
            # Verify installation
            try {
                python -c "import isaacsim" 2>$null
            }
            catch {
                Write-Error "ERROR: Isaac Sim installation verification failed"
                exit 1
            }
        }
        finally {
            Remove-Item -Path $tempDir -Recurse -Force -ErrorAction SilentlyContinue
        }
        
        # Fix cryptography conflict
        if ($env:ISAAC_PATH) {
            $cryptographyPath = Join-Path $env:ISAAC_PATH "exts\omni.pip.cloud\pip_prebundle\cryptography"
            if (Test-Path $cryptographyPath) {
                Write-Host "Fixing cryptography conflict..."
                Remove-Item -Path $cryptographyPath -Recurse -Force
            }
        }
    }
    
    # Install datasets
    if ($Dataset) {
        Write-Host "Installing datasets..."
        
        # Determine if we should accept dataset license automatically
        $DatasetAcceptFlag = "False"
        if ($AcceptDatasetTos) {
            $DatasetAcceptFlag = "True"
        }
        
        $pythonScript = @"
import os
os.environ['OMNI_KIT_ACCEPT_EULA'] = 'YES'
try:
    from omnigibson.macros import gm
    from omnigibson.utils.asset_utils import download_assets, download_og_dataset
    
    dataset_exists = os.path.exists(gm.DATASET_PATH)
    assets_exist = os.path.exists(gm.ASSET_PATH)
    
    if not (dataset_exists and assets_exist):
        print(f'Installing data to:')
        print(f'  Dataset (~25GB): {gm.DATASET_PATH}')
        print(f'  Assets (~2.5GB): {gm.ASSET_PATH}')
        
        if not dataset_exists:
            print('Downloading dataset...')
            download_og_dataset(accept_license=$DatasetAcceptFlag)
        
        if not assets_exist:
            print('Downloading assets...')
            download_assets()
    else:
        print('Datasets already exist, skipping download.')
except Exception as e:
    print(f'ERROR: Dataset installation failed: {e}')
    exit(1)
"@
        
        python -c $pythonScript
    }
    
    Write-Host "OmniGibson installation completed successfully!"
}

# Install JoyLo
if ($Teleop) {
    Write-Host "Installing JoyLo..."
    
    if (-not (Test-Path "joylo")) {
        Write-Error "ERROR: joylo directory not found"
        exit 1
    }
    
    pip install -e "$WorkDir\joylo"
}

if ($Eval) {
    Write-Host "Installing evaluation dependencies..."

    # get torch version via pip
    $TorchVersion = (pip show torch | Select-String "Version" | ForEach-Object { $_.ToString().Split(" ")[-1] })
    pip install torch-cluster -f "https://data.pyg.org/whl/torch-$TorchVersion.html"
    conda install av -c conda-forge -y
}

# Installation summary
Write-Host ""
Write-Host "=== Installation Complete! ==="
if ($NewEnv) { Write-Host "✓ Created conda environment 'behavior'" }
if ($OmniGibson) { Write-Host "✓ Installed OmniGibson + Isaac Sim" }
if ($BDDL) { Write-Host "✓ Installed BDDL" }
if ($Teleop) { Write-Host "✓ Installed JoyLo" }
if ($Primitives) { Write-Host "✓ Installed primitives support" }
if ($Eval) { Write-Host "✓ Installed evaluation support" }
if ($Dataset) { Write-Host "✓ Downloaded datasets" }
Write-Host ""
if ($NewEnv) { Write-Host "To activate: conda activate behavior" }