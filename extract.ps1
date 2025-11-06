$zipPath = "C:\Users\richa\Downloads\Larth-Etruscan-NLP-main.zip"
$destPath = "C:\etruscan"

# Make sure destination exists
New-Item -ItemType Directory -Force -Path $destPath | Out-Null

Add-Type -AssemblyName System.IO.Compression.FileSystem

# Open the zip
$zip = [System.IO.Compression.ZipFile]::OpenRead($zipPath)

foreach ($entry in $zip.Entries) {
    # Replace illegal chars (: * ? " < > |) with underscore
    $safeName = ($entry.FullName -replace '[:*?"<>|]', '_')

    $targetFile = Join-Path $destPath $safeName
    $targetDir  = Split-Path $targetFile -Parent

    if (-not (Test-Path $targetDir)) {
        New-Item -ItemType Directory -Force -Path $targetDir | Out-Null
    }

    if ($entry.Name -ne "") {
        [System.IO.Compression.ZipFileExtensions]::ExtractToFile($entry, $targetFile, $true)
    }
}

$zip.Dispose()
Write-Host "Extraction complete. Files are in $destPath"
