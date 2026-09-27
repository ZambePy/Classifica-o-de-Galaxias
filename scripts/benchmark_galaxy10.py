"""Benchmark no Galaxy10 DECaLS - dataset publico, resultados comparaveis.

    # baixe primeiro (2,6 GB):
    #   curl -L -o C:/astro-data/external/Galaxy10_DECals.h5 \
    #     https://astro.utoronto.ca/~hleung/shared/Galaxy10/Galaxy10_DECals.h5

    python scripts/benchmark_galaxy10.py --task 10class   # benchmark
    python scripts/benchmark_galaxy10.py --task 3class    # troca de survey

FECHA DOIS ITENS DO ROTEIRO, E POR UM MOTIVO SO: o Galaxy10 DECaLS E imagem
do DECaLS. Entao o mesmo arquivo responde as duas perguntas.

  --task 10class   "Benchmark contra trabalhos publicados"

      Treina na taxonomia OFICIAL do Galaxy10 (10 classes) com o nosso
      backbone, nossas transformacoes e nosso laco de treino. O numero que
      sai e diretamente comparavel com qualquer artigo que use este dataset,
      porque o dataset e a tarefa sao identicos. E o unico jeito honesto de
      responder "esse pipeline presta?" - a nossa acuracia de 3 classes nao
      se compara com nada publicado, ja que a taxonomia e nossa.

  --task 3class    "Trocar o survey das galaxias para DECaLS"

      Mapeia as 10 classes nas nossas 3 e treina. Produz um modelo de nivel 2
      treinado em DECaLS em vez de SDSS/Galaxy Zoo. E, de graca, o
      experimento que interessa mais: avaliar o modelo treinado em SDSS
      NESTE conjunto de teste do DECaLS mede TRANSFERENCIA ENTRE
      LEVANTAMENTOS - quanto do que a rede aprendeu era o objeto e quanto era
      a assinatura do instrumento.

O CAVEAT QUE PRECISA ESTAR NO TEXTO

O Galaxy10 DECaLS vem da campanha Galaxy Zoo DECaLS; nossas galaxias vem do
Galaxy Zoo 2 (imagens SDSS, via o desafio do Kaggle). As duas campanhas
rotularam ceu que se sobrepoe. O arquivo .h5 traz `ra`/`dec`, mas as imagens
do Kaggle vem so com `GalaxyID`, sem coordenada publica - entao NAO da para
excluir objetos em comum por cruzamento de posicao.

Consequencia: no teste de transferencia, parte dos objetos do conjunto de
teste do DECaLS pode ser o mesmo objeto que o modelo viu em SDSS, fotografado
por outro telescopio. Isso torna o numero de transferencia OTIMISTA. Esta
limitacao e real, nao da para contornar com os dados que temos, e tem de
aparecer junto do resultado.

O SEGUNDO CAVEAT, QUE EMPURRA PARA O OUTRO LADO

As taxonomias nao sao a mesma coisa com nomes diferentes. Medido:

    irregular no Galaxy Zoo 2 (nosso)         6% das galaxias
    irregular no Galaxy10 (Disturbed+Merging) 17% das galaxias

"Perturbada ou em fusao" e uma categoria mais larga que a nossa `irregular`.
Entao o teste de transferencia mistura DUAS mudancas: o levantamento (SDSS ->
DECaLS) e a definicao das classes. Um modelo treinado com 6% de irregular e
avaliado num conjunto com 17% perde acuracia por desvio de prior, nao por
desvio de dominio.

Por isso o relatorio traz ACURACIA BALANCEADA e macro-F1 junto da acuracia
crua: as duas primeiras pesam as classes igualmente e nao se movem quando
apenas a proporcao muda. Ao ler a queda, olhe a acuracia balanceada - a crua
mistura os dois efeitos e exagera o resultado.

Referencias:
  Leung & Bovy (2019), astroNN - o pacote que distribui o Galaxy10.
  Walmsley et al. (2022), Galaxy Zoo DECaLS, MNRAS 509, 3966.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]

H5_PADRAO = "C:/astro-data/external/Galaxy10_DECals.h5"

# Taxonomia oficial do Galaxy10 DECaLS, na ordem dos indices do arquivo.
CLASSES_10 = [
    "Disturbed",
    "Merging",
    "Round Smooth",
    "In-between Round Smooth",
    "Cigar Shaped Smooth",
    "Barred Spiral",
    "Unbarred Tight Spiral",
    "Unbarred Loose Spiral",
    "Edge-on without Bulge",
    "Edge-on with Bulge",
]

# Mapa das 10 classes do Galaxy10 para as 3 da nossa taxonomia.
#
# As decisoes discutiveis, explicitadas:
#
#   Cigar Shaped Smooth (4) -> elliptical. E "smooth", sem disco nem bracos;
#   a forma alongada e uma eliptica vista de lado (tipo E6/E7).
#
#   Edge-on, 8 e 9 -> spiral. Sao galaxias de DISCO vistas de perfil, onde
#   nao da para ver se ha bracos. No Galaxy Zoo 2 - de onde vem o nosso
#   rotulo `spiral` - o ramo "features or disk" e o que leva a espiral, e
#   edge-on esta dentro desse ramo. Sao 3296 imagens (19% do dataset), entao
#   a escolha pesa: use --edge-on exclude para medir sem elas.
#
#   Disturbed (0) e Merging (1) -> irregular. Galaxias perturbadas e em fusao
#   sao o que a nossa classe `irregular` captura no Galaxy Zoo.
MAPA_3 = {
    0: "irregular",
    1: "irregular",
    2: "elliptical",
    3: "elliptical",
    4: "elliptical",
    5: "spiral",
    6: "spiral",
    7: "spiral",
    8: "spiral",
    9: "spiral",
}
EDGE_ON = (8, 9)


def preparar_memmap(caminho_h5: Path, progress=print):
    """Converte as imagens do .h5 num .npy memory-mapped, uma vez.

    ISTO NAO E OTIMIZACAO PREMATURA - SEM ISTO O BENCHMARK E INVIAVEL.

    O Galaxy10_DECals.h5 guarda `images` com `chunks=(555, 16, 16, 1)` e
    compressao gzip. Esse layout e otimo para ler "o pixel (x,y) de todas as
    imagens" e PESSIMO para ler "uma imagem inteira", que e exatamente o que um
    DataLoader faz.

    A conta: uma imagem tem 256x256x3. Em chunks de 16x16x1, ela se espalha por
    (256/16) x (256/16) x 3 = 768 chunks. E cada chunk contem a mesma regiao de
    555 imagens diferentes. Ou seja, para extrair 196 KB o h5py descomprime
    768 x 555 x 256 bytes = ~109 MB. Amplificacao de ~560x.

    Medido nesta maquina:

        leitura imagem por imagem      1,7 img/s   -> 120 min por epoca
        leitura em fatia de 555      338,0 img/s   ->   0,9 min o dataset todo

    200x de diferenca, e a primeira versao deste script usava a leitura lenta:
    o treino ficava com a GPU a 6% de uso, esperando disco, e uma rodada de 12
    epocas levaria mais de 20 horas.

    A conversao le em fatias ALINHADAS AO CHUNK, entao cada chunk e
    descomprimido uma vez e serve as 555 imagens que ele contem. O resultado vai
    para um .npy de 3,5 GB lido com `mmap_mode='r'`: acesso aleatorio passa a
    ser paginacao do sistema operacional, nao descompressao.
    """
    import h5py
    import numpy as np

    destino = caminho_h5.with_suffix(".images.npy")

    with h5py.File(caminho_h5, "r") as f:
        forma = f["images"].shape
        chunks = f["images"].chunks

    if destino.exists():
        try:
            m = np.load(destino, mmap_mode="r")
            if m.shape == forma:
                progress(f"cache de imagens ja existe: {destino.name}")
                return m
            progress(f"cache com forma {m.shape} != {forma}; refazendo")
        except (ValueError, OSError) as exc:
            progress(f"cache ilegivel ({exc}); refazendo")

    gb = int(np.prod(forma)) / 1e9
    progress(
        f"preparando cache de imagens ({gb:.1f} GB em {destino.name}).\n"
        f"  O .h5 usa chunks={chunks}, que tornam a leitura imagem-por-imagem\n"
        f"  ~200x mais lenta que em fatia. Convertido uma vez, o treino deixa de\n"
        f"  esperar disco. Leva cerca de um minuto."
    )

    # Fatia alinhada ao chunk na primeira dimensao: cada chunk e lido uma vez.
    passo = chunks[0] if chunks else 512
    saida = np.lib.format.open_memmap(destino, mode="w+", dtype=np.uint8, shape=forma)
    with h5py.File(caminho_h5, "r") as f:
        origem = f["images"]
        for inicio in range(0, forma[0], passo):
            fim = min(inicio + passo, forma[0])
            saida[inicio:fim] = origem[inicio:fim]
            progress(f"  {fim}/{forma[0]}")
    saida.flush()
    del saida
    return np.load(destino, mmap_mode="r")


JANELA_LOCK_S = 900  # 15 min sem atualizacao = lock abandonado


def checar_loss_finita(valor: float, epoca: int, lote: int) -> None:
    """Aborta no primeiro NaN ou infinito na loss. ESTA GUARDA FALTAVA.

    Uma rodada desta mesma tarefa divergiu para NaN na epoca 3 e seguiu
    treinando ate a 7, com a acuracia de validacao travada em 0,5525 - que e
    exatamente a fracao da classe majoritaria do Galaxy10 em 3 classes. Ou seja:
    o modelo passou a prever sempre `spiral`.

    Nada avisou. A loss virou `nan`, o laco continuou, o early stopping
    "funcionou" (7 epocas sem melhora), e o JSON final saiu com aparencia
    normal. Foi o mesmo sintoma do primeiro bug silencioso do projeto - acuracia
    0,567 prevendo sempre a mesma classe - e o projeto ja tinha uma guarda para
    isso em `training.loops.check_numerical_sanity`, que este script nao usava
    porque tem o seu proprio laco de treino. Escrever um laco novo custou
    reescrever as protecoes: e a licao.

    `not (valor == valor)` detecta NaN sem depender de math.isnan; o teste de
    `abs(valor) == inf` pega o overflow, que normalmente vem antes do NaN.
    """
    if not (valor == valor) or abs(valor) == float("inf"):
        raise SystemExit(
            f"\n[ABORTADO] a loss virou {valor} na epoca {epoca}, lote {lote}.\n"
            f"  Treinar a partir daqui produz um modelo que preve sempre a classe\n"
            f"  majoritaria, e as metricas finais NAO avisam isso.\n\n"
            f"  Neste projeto o NaN veio de: precisao mista nesta GPU (ver o\n"
            f"  comentario em configs/level1_object.yaml), lote grande demais, ou\n"
            f"  learning rate alto. Tente --lr ou --batch-size menores."
        )


def checar_lock(destino_ckpt: Path, force: bool) -> Path:
    """Impede dois processos de treinarem no MESMO arquivo de checkpoint.

    ACONTECEU, E QUASE CORROMPEU UM TREINO DE UMA HORA.

    Uma cadeia de scripts rodava `--task 3class` enquanto eu rodava o mesmo
    comando a mao. Os dois gravam `galaxy10_3class.pt` quando a validacao
    melhora. Duas chamadas de `torch.save` no mesmo caminho ao mesmo tempo nao
    dao erro: uma trunca a outra, e o arquivo resultante pode nao carregar - ou
    pior, carregar pesos de uma mistura das duas.

    O projeto ja teve esse bug por outro caminho (um experimento sobrescrevendo
    o checkpoint de producao), e a correcao de la foi uma regra de nomes. Esta e
    a parte que faltava: protecao contra CONCORRENCIA.

    O lock guarda o PID e e atualizado a cada gravacao. Um lock mais velho que
    JANELA_LOCK_S e considerado abandonado (processo morto sem limpar) e nao
    bloqueia - senao um crash exigiria remover arquivo a mao.
    """
    import os
    import time

    lock = destino_ckpt.with_suffix(".lock")
    if lock.exists():
        idade = time.time() - lock.stat().st_mtime
        try:
            dono = lock.read_text(encoding="utf-8").strip()
        except (OSError, UnicodeDecodeError):
            # UnicodeDecodeError e real: um lock truncado no meio de uma
            # gravacao vira bytes invalidos, e o script morria aqui em vez de
            # cumprir o seu proposito, que e justamente proteger de gravacao
            # concorrente. Um teste deste arquivo cobre o caso.
            dono = "?"

        if idade >= JANELA_LOCK_S:
            # Processo morreu sem limpar. Nao bloqueia: exigir remocao manual
            # depois de um crash so treina a pessoa a apagar locks sem ler.
            print(
                f"[aviso] lock abandonado, ignorando: {lock.name} "
                f"(PID {dono}, parado ha {idade / 60:.0f} min)"
            )
        elif force:
            # Mensagem separada de proposito: a primeira versao deste codigo
            # caia no aviso de "lock antigo" tambem aqui, e imprimia coisas como
            # "lock antigo ignorado (0s > 900s)" - texto que se contradiz.
            print(
                f"[aviso] --force: ignorando lock ATIVO de {idade:.0f}s "
                f"(PID {dono}). Se aquele treino ainda roda, os dois vao gravar "
                f"em {destino_ckpt.name} e podem se corromper."
            )
        else:
            raise SystemExit(
                f"\n[ABORTADO] outro processo parece estar treinando neste mesmo\n"
                f"           checkpoint: {destino_ckpt.name}\n"
                f"           lock: {lock.name} (PID {dono}, atualizado ha {idade:.0f}s)\n\n"
                f"  Dois treinos gravando o mesmo arquivo podem corromper um ao outro.\n"
                f"  Opcoes:\n"
                f"    --tag <nome>   grava em galaxy10_*_<nome>.pt, sem colidir\n"
                f"    --force        assume o risco (use se o outro processo morreu)\n"
            )

    lock.parent.mkdir(parents=True, exist_ok=True)
    lock.write_text(str(os.getpid()), encoding="utf-8")
    return lock


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--h5", default=H5_PADRAO)
    parser.add_argument("--task", choices=["10class", "3class"], default="10class")
    parser.add_argument("--device", default=None)
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--backbone", default="resnet50")
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--force",
        action="store_true",
        help="ignora o lock de outro treino no mesmo checkpoint. Ver checar_lock.",
    )
    parser.add_argument(
        "--tag",
        default="",
        help="sufixo para o checkpoint e o JSON de saida. Serve para rodar uma "
        "variante sem sobrescrever a anterior (ex.: --tag v2 grava "
        "galaxy10_10class_v2.pt e 10class_v2.json).",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=0,
        help="workers do DataLoader. 0 basta com o cache memmap; ver preparar_memmap",
    )
    parser.add_argument(
        "--edge-on",
        choices=["spiral", "exclude"],
        default="spiral",
        help="so para --task 3class: o que fazer com as galaxias de perfil",
    )
    parser.add_argument(
        "--eval-only",
        action="store_true",
        help="nao treina; so avalia os checkpoints que ja existem",
    )
    args = parser.parse_args()

    # Pesados aqui dentro por causa do `spawn` do Windows.
    import h5py
    import numpy as np
    import torch
    import torch.nn as nn
    from sklearn.metrics import (
        accuracy_score,
        balanced_accuracy_score,
        classification_report,
        f1_score,
        matthews_corrcoef,
    )
    from torch.utils.data import DataLoader, Dataset

    from astro_classifier.config import resolve_device
    from astro_classifier.data.transforms import eval_transforms, train_transforms
    from astro_classifier.models.backbone import build_backbone
    from astro_classifier.paths import get_paths
    from astro_classifier.taxonomy import Level

    caminho_h5 = Path(args.h5)
    if not caminho_h5.exists():
        print(f"Arquivo nao encontrado: {caminho_h5}")
        print("\nBaixe com (2,6 GB):")
        print("  curl -L -o C:/astro-data/external/Galaxy10_DECals.h5 \\")
        print("    https://astro.utoronto.ca/~hleung/shared/Galaxy10/Galaxy10_DECals.h5")
        return 1

    device = resolve_device(args.device)
    paths = get_paths().ensure()
    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)

    with h5py.File(caminho_h5, "r") as f:
        y10 = f["ans"][:].astype(np.int64)
    print(f"{caminho_h5.name}: {len(y10)} imagens, 10 classes\n")

    # ------------------------------------------------- tarefa e rotulos
    if args.task == "10class":
        classes = CLASSES_10
        indices = np.arange(len(y10))
        y = y10.copy()
        titulo = "GALAXY10 DECALS - taxonomia oficial de 10 classes"
    else:
        classes = list(Level.GALAXY.classes)  # elliptical, spiral, irregular
        ordem = {c: i for i, c in enumerate(classes)}
        mantidas = (
            np.array([c for c in range(10) if c not in EDGE_ON])
            if args.edge_on == "exclude"
            else np.arange(10)
        )
        indices = np.where(np.isin(y10, mantidas))[0]
        y = np.array([ordem[MAPA_3[int(c)]] for c in y10[indices]], dtype=np.int64)
        titulo = f"GALAXY10 DECALS -> nossas 3 classes (edge-on: {args.edge_on})"
        if args.edge_on == "exclude":
            print(f"excluidas {len(y10) - len(indices)} imagens edge-on\n")

    print("=" * 72)
    print(titulo)
    print("=" * 72)
    for i, c in enumerate(classes):
        print(f"  {i} {c:<26} {int((y == i).sum()):>6}")
    print()

    # ------------------------------------------------- split estratificado
    # 70/15/15, estratificado, semente fixa. Os artigos que usam o Galaxy10
    # nao compartilham um split oficial, entao o nosso vai declarado aqui e o
    # arquivo de resultados guarda a semente.
    treino_idx, val_idx, teste_idx = [], [], []
    for c in range(len(classes)):
        pos = np.where(y == c)[0]
        rng.shuffle(pos)
        n_tr = int(0.70 * len(pos))
        n_va = int(0.15 * len(pos))
        treino_idx += list(pos[:n_tr])
        val_idx += list(pos[n_tr : n_tr + n_va])
        teste_idx += list(pos[n_tr + n_va :])
    treino_idx, val_idx, teste_idx = map(np.array, (treino_idx, val_idx, teste_idx))
    print(f"split  treino {len(treino_idx)}  val {len(val_idx)}  teste {len(teste_idx)}\n")

    memmap = preparar_memmap(caminho_h5)

    class Galaxy10(Dataset):
        """Le do cache .npy memory-mapped preparado por `preparar_memmap`.

        NAO le do .h5 diretamente, e a razao esta documentada em
        `preparar_memmap`: o layout de chunks daquele arquivo torna a leitura
        de uma imagem isolada 200x mais lenta que a leitura em fatia.
        """

        def __init__(self, posicoes, transform):
            self.posicoes = posicoes  # posicoes em `indices`/`y`
            self.transform = transform

        def __len__(self):
            return len(self.posicoes)

        def __getitem__(self, i):
            from PIL import Image

            p = int(self.posicoes[i])
            bruto = memmap[int(indices[p])]  # (256,256,3) uint8
            return self.transform(Image.fromarray(bruto)), int(y[p])

    t_treino = train_transforms(args.image_size, augment=True)
    t_eval = eval_transforms(args.image_size)

    def carregar(posicoes, transform, shuffle):
        # num_workers=0 e o padrao de proposito: com o cache memmap a leitura ja
        # e paginacao do SO, e o gargalo volta a ser a GPU. Workers aqui so
        # adicionariam memoria comprometida no Windows (`spawn` reimporta torch
        # em cada processo, ~600 MB) sem ganho. Use --workers se o perfil mudar.
        return DataLoader(
            Galaxy10(posicoes, transform),
            batch_size=args.batch_size,
            shuffle=shuffle,
            num_workers=args.workers,
        )

    sufixo = f"_{args.tag}" if args.tag else ""
    destino_ckpt = paths.checkpoints / f"galaxy10_{args.task}{sufixo}.pt"
    # So trava quando for TREINAR: --eval-only nao grava nada.
    lock = None if args.eval_only else checar_lock(destino_ckpt, args.force)
    if lock is not None:
        # `atexit` em vez de remover na ultima linha: garante a limpeza tambem
        # quando o treino aborta (NaN, Ctrl-C, excecao). Ver `liberar_lock`.
        import atexit

        atexit.register(liberar_lock, destino_ckpt)

    # ------------------------------------------------- modelo
    class Rede(nn.Module):
        def __init__(self):
            super().__init__()
            self.backbone, n = build_backbone(args.backbone, pretrained=True)
            self.head = nn.Sequential(nn.Dropout(0.2), nn.Linear(n, len(classes)))

        def forward(self, x):
            return self.head(self.backbone(x).flatten(1))

    modelo = Rede().to(device)

    @torch.inference_mode()
    def prever(loader):
        modelo.eval()
        ys, ps = [], []
        for x, alvo in loader:
            ps.append(modelo(x.to(device)).argmax(1).cpu().numpy())
            ys.append(alvo.numpy())
        return np.concatenate(ys), np.concatenate(ps)

    if not args.eval_only:
        loader_tr = carregar(treino_idx, t_treino, True)
        loader_va = carregar(val_idx, t_eval, False)

        otim = torch.optim.AdamW(modelo.parameters(), lr=args.lr, weight_decay=1e-4)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(otim, T_max=args.epochs)
        # mixed precision DESLIGADO: ver o comentario em configs/level1_object.yaml.
        criterio = nn.CrossEntropyLoss(label_smoothing=0.05)

        melhor, melhor_ep, sem_melhora = -1.0, 0, 0
        historico = []
        print(f"treinando em {device}, {args.epochs} epocas\n")
        for ep in range(1, args.epochs + 1):
            modelo.train()
            t0, soma, n = time.time(), 0.0, 0
            for x, alvo in loader_tr:
                x, alvo = x.to(device, non_blocking=True), alvo.to(device, non_blocking=True)
                otim.zero_grad(set_to_none=True)
                perda = criterio(modelo(x), alvo)
                perda.backward()
                otim.step()

                valor = float(perda.detach())
                checar_loss_finita(valor, ep, n + 1)
                soma += valor
                n += 1
            sched.step()

            yv, pv = prever(loader_va)
            acc = accuracy_score(yv, pv)
            f1 = f1_score(yv, pv, average="macro")
            historico.append({"epoca": ep, "loss": soma / n, "val_acc": acc, "val_macro_f1": f1})
            print(
                f"epoca {ep:>2}/{args.epochs}  loss {soma / n:.4f}  "
                f"val_acc {acc:.4f}  macro-F1 {f1:.4f}  {time.time() - t0:.0f}s"
            )

            if acc > melhor:
                melhor, melhor_ep, sem_melhora = acc, ep, 0
                torch.save(
                    {
                        "state_dict": modelo.state_dict(),
                        "classes": classes,
                        "backbone": args.backbone,
                        "task": args.task,
                        "edge_on": args.edge_on,
                        "seed": args.seed,
                        "val_acc": acc,
                        "epoca": ep,
                    },
                    destino_ckpt,
                )
                print(f"  -> melhor salvo em {destino_ckpt}")
                if lock is not None:
                    lock.touch()  # mantem o lock fresco enquanto o treino vive
            else:
                sem_melhora += 1
                if sem_melhora >= 5:
                    print("  -> early stopping")
                    break
        print(f"\nmelhor validacao: {melhor:.4f} (epoca {melhor_ep})")
    else:
        # --eval-only NAO treina, entao nao tem historico proprio - mas tambem
        # nao pode APAGAR o que ja existe. A primeira versao gravava lista vazia
        # e destruia a procedencia do treino anterior: um `--eval-only` inocente
        # (rodado so para conferir uma metrica) apagava as 12 epocas do JSON, e o
        # arquivo passava a nao dizer mais como aquele modelo foi treinado.
        #
        # Recuperar deu para fazer porque o log ainda existia. Nao e garantido.
        historico = []
        anterior = paths.runs / "_galaxy10" / f"{args.task}{sufixo}.json"
        if anterior.exists():
            try:
                historico = json.loads(anterior.read_text(encoding="utf-8")).get(
                    "historico", []
                )
                if historico:
                    print(f"historico do treino preservado de {anterior.name} "
                          f"({len(historico)} epocas)")
            except (OSError, json.JSONDecodeError) as exc:
                print(f"[aviso] nao consegui ler o historico anterior: {exc}")

    if not destino_ckpt.exists():
        print(f"\nCheckpoint nao encontrado: {destino_ckpt}")
        return 1

    ck = torch.load(destino_ckpt, map_location=device, weights_only=False)
    if ck["classes"] != classes:
        print(f"[ERRO] o checkpoint foi treinado em {ck['classes']}, a tarefa atual e {classes}")
        return 1
    modelo.load_state_dict(ck["state_dict"])
    modelo.to(device).eval()

    # ------------------------------------------------- teste
    print("\n" + "=" * 72)
    print("CONJUNTO DE TESTE")
    print("=" * 72)
    yt, pt = prever(carregar(teste_idx, t_eval, False))
    metricas = {
        "acuracia": float(accuracy_score(yt, pt)),
        "acuracia_balanceada": float(balanced_accuracy_score(yt, pt)),
        "macro_f1": float(f1_score(yt, pt, average="macro")),
        "mcc": float(matthews_corrcoef(yt, pt)),
        "n_teste": int(len(yt)),
    }
    print(f"\n  acuracia             {metricas['acuracia']:.4f}")
    print(f"  acuracia balanceada  {metricas['acuracia_balanceada']:.4f}")
    print(f"  macro-F1             {metricas['macro_f1']:.4f}")
    print(f"  MCC                  {metricas['mcc']:.4f}")
    print(f"  n                    {metricas['n_teste']}")
    print()
    print(classification_report(yt, pt, target_names=classes, digits=3, zero_division=0))

    resultado = {
        "tarefa": args.task,
        "backbone": args.backbone,
        "classes": classes,
        "seed": args.seed,
        "split": {"treino": len(treino_idx), "val": len(val_idx), "teste": len(teste_idx)},
        "edge_on": args.edge_on if args.task == "3class" else None,
        "teste": metricas,
        "historico": historico,
    }

    if args.task == "10class":
        print(
            "COMO COMPARAR COM A LITERATURA\n"
            "------------------------------\n"
            "  A acuracia acima e na taxonomia oficial de 10 classes, entao e\n"
            "  comparavel com trabalhos que usam este dataset. Confira o numero\n"
            "  de referencia na documentacao do astroNN (Leung & Bovy 2019), que\n"
            "  distribui o Galaxy10, e cite a versao DECaLS - existe tambem uma\n"
            "  versao SDSS mais antiga, com acuracias diferentes, e confundir as\n"
            "  duas e o erro classico aqui.\n"
            "\n"
            "  O split NAO e oficial: nao ha um publicado. O nosso e 70/15/15\n"
            "  estratificado com semente fixa, registrado neste JSON. Uma\n"
            "  diferenca de 1 a 2 pontos entre trabalhos pode ser so isso."
        )

    # --------------------------- transferencia entre levantamentos (3 classes)
    if args.task == "3class":
        print("\n" + "=" * 72)
        print("TRANSFERENCIA ENTRE LEVANTAMENTOS - modelo treinado em SDSS, testado em DECaLS")
        print("=" * 72)
        sdss_ckpt = paths.checkpoints / "galaxy_best.pt"
        if not sdss_ckpt.exists():
            print(f"  {sdss_ckpt} nao existe - treine o nivel 2 antes.")
        else:
            from astro_classifier.models.classifier import AstroClassifier

            sdss, _ = AstroClassifier.load(sdss_ckpt, device=device)
            # As classes batem por construcao: os dois usam Level.GALAXY.classes.
            @torch.inference_mode()
            def prever_sdss(loader):
                sdss.eval()
                ys, ps = [], []
                for x, alvo in loader:
                    ps.append(sdss(x.to(device)).argmax(1).cpu().numpy())
                    ys.append(alvo.numpy())
                return np.concatenate(ys), np.concatenate(ps)

            yx, px = prever_sdss(carregar(teste_idx, t_eval, False))
            cross = {
                "acuracia": float(accuracy_score(yx, px)),
                "acuracia_balanceada": float(balanced_accuracy_score(yx, px)),
                "macro_f1": float(f1_score(yx, px, average="macro")),
                "mcc": float(matthews_corrcoef(yx, px)),
            }
            resultado["transferencia_sdss_para_decals"] = cross

            print(f"\n  {'':<24}{'DECaLS':>10}{'SDSS':>10}{'queda':>9}")
            print("  " + "-" * 53)
            for k, rot in [
                ("acuracia", "acuracia"),
                ("acuracia_balanceada", "acuracia balanceada"),
                ("macro_f1", "macro-F1"),
                ("mcc", "MCC"),
            ]:
                a, b = metricas[k], cross[k]
                print(f"  {rot:<24}{a:>10.4f}{b:>10.4f}{b - a:>+9.4f}")

            print(
                "\n  A coluna SDSS e o modelo do nivel 2 - treinado em Galaxy Zoo 2,\n"
                "  imagem SDSS - avaliado nestas imagens do DECaLS. A queda mede\n"
                "  quanto do desempenho dependia do instrumento e nao do objeto.\n"
                "\n"
                "  DOIS CAVEATS, QUE EMPURRAM EM DIRECOES OPOSTAS:\n"
                "\n"
                "  1. As duas campanhas do Galaxy Zoo cobrem ceu sobreposto, e nao\n"
                "     temos coordenada das imagens do Kaggle - objetos em comum nao\n"
                "     puderam ser removidos. Isso torna a coluna SDSS OTIMISTA.\n"
                "\n"
                "  2. As taxonomias diferem: `irregular` e 6% das nossas galaxias e\n"
                "     17% das do Galaxy10. Parte da queda e desvio de PRIOR, nao de\n"
                "     dominio. Isso torna a queda na acuracia crua PESSIMISTA.\n"
                "\n"
                "  Olhe a acuracia BALANCEADA: ela e imune ao segundo efeito."
            )

    saida = paths.runs / "_galaxy10"
    saida.mkdir(parents=True, exist_ok=True)
    destino_json = saida / f"{args.task}{sufixo}.json"
    destino_json.write_text(
        json.dumps(resultado, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(f"\nResultados em {destino_json}")
    return 0


def liberar_lock(destino_ckpt: Path) -> bool:
    """Remove o lock do checkpoint. Devolve se havia algo para remover.

    CHAMADO EM `finally`, NAO NO FIM DO SUCESSO.

    A primeira versao removia o lock na ultima linha de `main()`, e isso tem dois
    furos. O menor: se o treino levantar excecao, o lock fica para tras e bloqueia
    a proxima tentativa por 15 minutos - exatamente quando a pessoa quer tentar de
    novo. O maior: aquela remocao nao chegou a existir. O patch que a adicionou
    falhou em silencio por um escape de `\\n` que nao casou, e ninguem notou
    porque nenhum teste cobria a limpeza - so a criacao.

    O lock orfao foi encontrado depois de uma rodada que terminou BEM: arquivo
    presente, PID de um processo morto, e a proxima execucao seria recusada sem
    motivo.
    """
    lock = destino_ckpt.with_suffix(".lock")
    try:
        lock.unlink()
        return True
    except FileNotFoundError:
        return False
    except OSError as exc:
        print(f"[aviso] nao consegui remover {lock.name}: {exc}")
        return False


if __name__ == "__main__":
    sys.exit(main())
