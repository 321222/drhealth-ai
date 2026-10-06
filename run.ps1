$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot

$port = 8502
while ($port -le 8599) {
    $listener = Get-NetTCPConnection -State Listen -LocalPort $port -ErrorAction SilentlyContinue
    if (-not $listener) {
        $probe = $null
        try {
            $probe = [System.Net.Sockets.TcpListener]::new(
                [System.Net.IPAddress]::Loopback,
                $port
            )
            $probe.Start()
            $probe.Stop()
            break
        }
        catch [System.Net.Sockets.SocketException] {
            if ($probe) {
                $probe.Stop()
            }
        }
    }
    $port++
}

if ($port -gt 8599) {
    throw "No available app port was found between 8502 and 8599."
}

$secretsDirectory = Join-Path $env:USERPROFILE ".streamlit"
$secretsFile = Join-Path $secretsDirectory "secrets.toml"
$hasStoredKey = $false
if (Test-Path $secretsFile) {
    $storedSettings = Get-Content -LiteralPath $secretsFile -Raw
    $hasStoredKey = $storedSettings -match "(?m)^\s*GEMINI_API_KEY\s*="
    $storedSettings = $null
}

$keyToSave = $null
if (-not $hasStoredKey -and $env:GEMINI_API_KEY) {
    $keyToSave = $env:GEMINI_API_KEY
}
elseif (-not $hasStoredKey -and [Console]::IsInputRedirected) {
    Write-Warning "No Gemini key is saved, and this session cannot accept hidden key input. Chat and Gemini voice will be unavailable until a key is saved with .\run.ps1 in an interactive PowerShell window."
}
elseif (-not $hasStoredKey) {
    $secureKey = Read-Host "Enter your Gemini API key (saved for this Windows account; input is hidden)" -AsSecureString
    $credential = [System.Net.NetworkCredential]::new("", $secureKey)
    $keyToSave = $credential.Password
}

if ($keyToSave) {
    $keyLiteral = ConvertTo-Json -InputObject $keyToSave -Compress
    $keyLine = "GEMINI_API_KEY = $keyLiteral"
    $utf8WithoutBom = [System.Text.UTF8Encoding]::new($false)
    New-Item -ItemType Directory -Path $secretsDirectory -Force | Out-Null

    if (Test-Path $secretsFile) {
        $existingSettings = [System.IO.File]::ReadAllText($secretsFile)
        $separator = if ($existingSettings -and $existingSettings[-1] -notin "`r", "`n") { "`r`n" } else { "" }
        [System.IO.File]::AppendAllText($secretsFile, "$separator$keyLine`r`n", $utf8WithoutBom)
    }
    else {
        [System.IO.File]::WriteAllText($secretsFile, "$keyLine`r`n", $utf8WithoutBom)
    }

    $keyLiteral = $null
    $keyLine = $null
    $keyToSave = $null
    $credential = $null
    if ($secureKey) {
        $secureKey.Dispose()
        $secureKey = $null
    }
}

if ($hasStoredKey -or (Test-Path -LiteralPath $secretsFile)) {
    $identity = [System.Security.Principal.WindowsIdentity]::GetCurrent()
    $fileAcl = Get-Acl -LiteralPath $secretsFile
    $userSid = $identity.User.Value
    $allowedSids = @(
        $fileAcl.Access | ForEach-Object {
            $_.IdentityReference.Translate(
                [System.Security.Principal.SecurityIdentifier]
            ).Value
        }
    )
    $aclNeedsUpdate = (
        -not $fileAcl.AreAccessRulesProtected -or
        $allowedSids.Count -ne 1 -or
        $allowedSids[0] -ne $userSid
    )

    if ($aclNeedsUpdate) {
        $fileAcl.SetAccessRuleProtection($true, $false)
        foreach ($accessRule in @($fileAcl.Access)) {
            $fileAcl.RemoveAccessRuleAll($accessRule)
        }
        $userRule = [System.Security.AccessControl.FileSystemAccessRule]::new(
            $identity.User,
            [System.Security.AccessControl.FileSystemRights]::FullControl,
            [System.Security.AccessControl.AccessControlType]::Allow
        )
        $fileAcl.SetAccessRule($userRule)
        Set-Acl -LiteralPath $secretsFile -AclObject $fileAcl
    }
}

Write-Host "Starting DrHealth AI at http://localhost:$port"
py -3.12 -m streamlit run .\Chatbot.py --server.port $port
$exitCode = $LASTEXITCODE

if ($exitCode -ne 0) {
    Write-Host "Streamlit stopped with exit code $exitCode. Review its output above for the cause."
    exit $exitCode
}
