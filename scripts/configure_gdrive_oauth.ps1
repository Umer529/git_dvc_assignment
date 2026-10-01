param(
    [Parameter(Mandatory = $true)]
    [string]$ClientId
)

$projectRoot = Split-Path -Parent $PSScriptRoot
$dvc = Join-Path $projectRoot "venv\Scripts\dvc.exe"

if (-not (Test-Path -LiteralPath $dvc)) {
    throw "DVC was not found at $dvc"
}

$secureSecret = Read-Host "Google OAuth client secret" -AsSecureString
$secretPointer = [Runtime.InteropServices.Marshal]::SecureStringToBSTR($secureSecret)

try {
    $clientSecret = [Runtime.InteropServices.Marshal]::PtrToStringBSTR($secretPointer)

    & $dvc remote modify --local gdrive_storage gdrive_client_id $ClientId
    if ($LASTEXITCODE -ne 0) { throw "Failed to save the OAuth client ID." }

    & $dvc remote modify --local gdrive_storage gdrive_client_secret $clientSecret
    if ($LASTEXITCODE -ne 0) { throw "Failed to save the OAuth client secret." }

    Write-Host "OAuth settings saved to ignored .dvc/config.local."
    Write-Host "Run: .\venv\Scripts\dvc.exe push"
}
finally {
    if ($secretPointer -ne [IntPtr]::Zero) {
        [Runtime.InteropServices.Marshal]::ZeroFreeBSTR($secretPointer)
    }
    $clientSecret = $null
}
