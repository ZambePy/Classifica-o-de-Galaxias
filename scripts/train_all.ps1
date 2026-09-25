# Treina e avalia os tres niveis da cascata, em sequencia, sem supervisao.
#
#   powershell -ExecutionPolicy Bypass -File scripts\train_all.ps1
#
# Pensado para rodar durante a noite:
#   - impede o Windows de suspender enquanto trabalha (sem precisar de admin)
#   - grava tudo num log com carimbo de hora, para leitura de manha
#   - a falha de um nivel NAO aborta os outros
#   - ordem do mais barato para o mais caro: se algo estiver errado na
#     configuracao, voce descobre em minutos, nao depois de duas horas
#
# Parametros uteis:
#   -SkipEval        so treina, nao avalia no conjunto de teste
#   -Levels nebula,galaxy    escolhe quais niveis rodar

param(
    [ValidateSet("nebula", "object", "galaxy")]
    [string[]]$Levels = @("nebula", "object", "galaxy"),
    [switch]$SkipEval
)

$ErrorActionPreference = "Continue"
$RepoRoot = Split-Path -Parent $PSScriptRoot
Set-Location $RepoRoot

$Python = Join-Path $RepoRoot ".venv\Scripts\python.exe"
if (-not (Test-Path $Python)) {
    Write-Host "Ambiente nao encontrado. Rode scripts\setup_env.ps1 antes." -ForegroundColor Red
    exit 1
}

# Do mais rapido para o mais lento.
$Configs = [ordered]@{
    nebula = "configs\level3_nebula.yaml"
    object = "configs\level1_object.yaml"
    galaxy = "configs\level2_galaxy.yaml"
}

$Stamp   = Get-Date -Format "yyyy-MM-dd_HHmm"
$LogDir  = "C:\astro-data\runs\_logs"
New-Item -ItemType Directory -Force -Path $LogDir | Out-Null
$LogFile = Join-Path $LogDir "train_all_$Stamp.log"

function Write-Log {
    param([string]$Text, [string]$Color = "Gray")
    $linha = "[{0}] {1}" -f (Get-Date -Format "HH:mm:ss"), $Text
    Write-Host $linha -ForegroundColor $Color
    Add-Content -Path $LogFile -Value $linha -Encoding utf8
}

# Impede suspensao enquanto o script roda. ES_CONTINUOUS | ES_SYSTEM_REQUIRED.
# A tela pode apagar; a maquina nao dorme. O estado volta ao normal sozinho
# quando este processo termina.
Add-Type -Namespace Energia -Name Win32 -MemberDefinition @'
[DllImport("kernel32.dll", SetLastError = true)]
public static extern uint SetThreadExecutionState(uint esFlags);
'@
[void][Energia.Win32]::SetThreadExecutionState(0x80000000 -bor 0x00000001)

Write-Log "=== TREINO DA CASCATA ===" "Cyan"
Write-Log "log: $LogFile"
Write-Log "niveis: $($Levels -join ', ')"
Write-Log "suspensao do Windows bloqueada enquanto este script rodar"
Write-Log ""

$Inicio = Get-Date
$Resultados = @()

foreach ($nivel in $Configs.Keys) {
    if ($Levels -notcontains $nivel) { continue }

    $config = $Configs[$nivel]
    Write-Log "--- treinando $nivel ($config) ---" "Cyan"
    $t0 = Get-Date

    & $Python scripts\train.py --config $config 2>&1 |
        Tee-Object -FilePath $LogFile -Append |
        Where-Object { $_ -notmatch "it/s\]|s/it\]" } |
        ForEach-Object { Write-Host $_ }

    $treinoOk = $LASTEXITCODE -eq 0
    $dur = [math]::Round(((Get-Date) - $t0).TotalMinutes, 1)

    if (-not $treinoOk) {
        Write-Log "FALHOU o treino de $nivel (codigo $LASTEXITCODE) apos $dur min" "Red"
        $Resultados += [pscustomobject]@{ Nivel = $nivel; Treino = "FALHOU"; Avaliacao = "-"; Minutos = $dur }
        continue
    }

    Write-Log "treino de $nivel concluido em $dur min" "Green"

    $avaliacao = "pulada"
    if (-not $SkipEval) {
        Write-Log "--- avaliando $nivel no conjunto de teste ---" "Cyan"
        & $Python scripts\evaluate.py --config $config 2>&1 |
            Tee-Object -FilePath $LogFile -Append |
            ForEach-Object { Write-Host $_ }
        $avaliacao = if ($LASTEXITCODE -eq 0) { "ok" } else { "FALHOU" }
    }

    $total = [math]::Round(((Get-Date) - $t0).TotalMinutes, 1)
    $Resultados += [pscustomobject]@{ Nivel = $nivel; Treino = "ok"; Avaliacao = $avaliacao; Minutos = $total }
    Write-Log ""
}

# Com os tres niveis treinados, mede o SISTEMA - nao os niveis isolados.
# E aqui que aparece o erro em cascata, que nenhuma metrica por nivel mostra.
$treinouTudo = ($Resultados | Where-Object { $_.Treino -eq "ok" }).Count -eq 3
if ($treinouTudo -and -not $SkipEval) {
    Write-Log "--- avaliando a CASCATA COMPLETA ---" "Cyan"
    & $Python scripts\evaluate_cascade.py 2>&1 |
        Tee-Object -FilePath $LogFile -Append |
        Where-Object { $_ -notmatch "it/s\]|s/it\]" } |
        ForEach-Object { Write-Host $_ }
    Write-Log ""
}

$TotalMin = [math]::Round(((Get-Date) - $Inicio).TotalMinutes, 1)

Write-Log "=== RESUMO ===" "Cyan"
$Resultados | Format-Table -AutoSize | Out-String -Width 100 |
    ForEach-Object { $_.TrimEnd() } | Where-Object { $_ } | ForEach-Object { Write-Log $_ }
Write-Log "tempo total: $TotalMin min"
Write-Log ""
Write-Log "De manha, olhe nesta ordem:"
Write-Log "  runs\_cascade\metrics.md          O SISTEMA funciona? erro em cascata"
Write-Log "  runs\_cascade\confusion.png       matriz sobre os rotulos finais"
Write-Log "  runs\<exp>\confusion_test.png     quem confunde com quem, por nivel"
Write-Log "  runs\<exp>\gradcam\<classe>.png   o modelo olhou o objeto ou o fundo?"
Write-Log "  runs\<exp>\curves.png             houve overfitting?"
Write-Log "  runs\<exp>\metrics_test.md        tabela pronta para o relatorio"
Write-Log "  $LogFile"

# Libera o bloqueio de suspensao.
[void][Energia.Win32]::SetThreadExecutionState(0x80000000)
