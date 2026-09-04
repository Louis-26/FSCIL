# Datasets

## 1. What the paper specifies (§4)

> "Following the experimental protocol of CLOSER, we evaluate our method on
> three widely used FSCIL benchmarks: miniImageNet, CIFAR-100 and CUB200.
> miniImageNet contains 100 classes with 60,000 images of size 84×84. CIFAR-100
> consists of 100 classes with 32×32 images, while CUB200 includes 200
> fine-grained bird categories with 224×224 resolution images. We adopt the
> class splits from [1]: for miniImageNet and CIFAR-100, 60 classes are used as
> base classes, and the remaining 40 are divided into eight 5-way 5-shot
> incremental sessions; for CUB200, 100 base classes are followed by ten 10-way
> 5-shot sessions."

## 2. What we actually use

| Dataset | Train | Test | Classes | Base | Sessions | Source |
|---|---|---|---|---|---|---|
| miniImageNet | 50,000 (500/cls) | 10,000 (100/cls) | 100 | 60 | 9 | `HarborYuan/Few-Shot-Class-Incremental-Learning` → `fscil.zip` |
| CIFAR-100 | 50,000 | 10,000 | 100 | 60 | 9 | cs.toronto.edu (official) |
| CUB-200-2011 | 5,994 | 5,794 | 200 | 100 | 11 | same `fscil.zip` |

Run `bash overview/data_prepare.sh` — it downloads, extracts, installs the
session index lists and runs the protocol self-test.

### Which miniImageNet? (this is the part that goes wrong)

"miniImageNet" names at least three incompatible datasets:

1. **Ravi & Larochelle few-shot cache** — `mini-imagenet-cache-{train,val,test}.pkl`,
   64/16/20 *class* split for episodic few-shot learning. **Not** an FSCIL
   benchmark.
2. **CEC / TOPIC FSCIL release** — all 100 classes, 500 train + 100 test images
   per class, original ImageNet resolution, split defined by
   `split/{train,test}.csv`. **This is what Table 1 uses.**
3. Assorted 84×84 or parquet re-uploads with unknown provenance and different
   per-image train/test assignment.

The link in this repository's old top-level README
(`drive.usercontent.google.com/...id=16V_ZlkW4SsnNDtnGmaBRq2OoPmUOc5mY`)
downloads `mini-imagenet.tar.gz`, which is **variant 1** — the few-shot cache.
It cannot reproduce Table 1. `data_prepare.sh` uses variant 2 and *proves* it:

```
cmp  data/miniimagenet/split/train.csv  complementary/index_list/mini_imagenet/train.csv
cmp  data/miniimagenet/split/test.csv   complementary/index_list/mini_imagenet/test.csv
```

Both are byte-identical (50,001 and 10,001 lines including headers), so the
downloaded images are in exactly the file-name space the CEC session lists
index into. The script aborts if they ever differ.

One deviation from the paper's prose worth noting: the CEC release stores
miniImageNet at **original ImageNet resolution** (typically 500×375), not
84×84. The 84×84 figure describes the historical Ravi & Larochelle cache.
Methods resize/crop to 84×84 in their transform. We keep the full-resolution
files and let each backbone apply its own preprocessing (CLIP → 224 centre
crop; ResNet-18 → 84 crop, matching CEC/CLOSER).

## 3. Class ordering

Label `c` is the position of the class in the benchmark's canonical ordering,
which for miniImageNet is **sorted WordNet-ID order** — asserted in
`data.build_mini_imagenet`. Base classes are `n01532829 … n03535780`
(labels 0–59); novel classes are `n03544143 … n13133613` (labels 60–99), split
into eight groups of five in that order. This matches
`complementary/index_list/mini_imagenet/process_miniImagenet.py` exactly.

Human-readable names come from `cdfscil/assets/mini_imagenet_classnames.json`,
built by `cdfscil/build_classnames.py` by cross-referencing the official
ImageNet class index (which the script asserts is in sorted-wnid order) with
`open_clip.IMAGENET_CLASSNAMES` (the names OpenAI used for CLIP's published
zero-shot numbers). Nothing is hand-typed.

## 4. Session composition

* **Session 0** = every train image of every base class
  (miniImageNet 30,000; CIFAR-100 30,000; CUB-200 3,000).
* **Session s ≥ 1** = exactly `way × shot` images listed in
  `index_list/<dataset>/session_{s+1}.txt` (miniImageNet/CIFAR 25; CUB 50).
* **Test at session s** = all test images of classes `0 … base + s·way − 1`.

`FSCILBenchmark.sanity_check()` asserts all of it — session-0 identity, per-class
shot counts, correct novel class sets, monotone test growth, and that the final
test set is the complete test split. It runs at the start of every script, so a
mis-prepared dataset fails immediately instead of producing plausible-looking
wrong numbers.

## 5. A caveat that matters for interpreting the results

miniImageNet's 100 classes are ImageNet-1k classes, and CLIP was pre-trained on
400M web image–text pairs. A frozen-CLIP method therefore arrives at the
"incremental" sessions already knowing the novel classes — CLIP zero-shot alone
scores 94.8 % → 91.8 % here. Every competitor in Table 1 trains a ResNet-18 from
scratch on the 60 base classes and has genuinely never seen the novel ones.

The two settings are not comparable, and this is not a subtle effect: it is
worth ~27 accuracy points at the last session. See `results.md` §3 and
`paper_discrepancies.md` §3.

## 6. LLM class descriptions: which datasets have them

The `llm` text mode reads `cdfscil/assets/descriptions/<dataset>.json`. Their
provenance differs, and each file records it in its own `_meta` block:

| dataset | descriptions | generator |
|---|---|---|
| **miniImageNet** | 6 per class, hand-verified to name their class | **Claude Opus 5**, single pass, prompt in `descriptions.GENERATION_PROMPT` |
| **CIFAR-100** | 6 per class, hand-verified to name their class | **Claude Opus 5**, same prompt |
| CUB-200 | 6 per class | deterministic **template fallback** (`generate_descriptions.py --provider template`) — the class name in six photo templates, no LLM |

miniImageNet and CIFAR-100 have genuine LLM-written visual descriptions. CUB-200
(200 fine-grained bird species) uses the committed template fallback, so its
`llm` rows should be read as "class-name prompts" rather than "LLM prior".
Regenerate it with a real LLM via

```bash
export ANTHROPIC_API_KEY=...
python -m cdfscil.generate_descriptions --dataset cub200 --provider anthropic --k 6
```

## 7. Licences

ImageNet/miniImageNet and CUB-200-2011 are research-use only; CIFAR-100 is
freely redistributable. The HuggingFace mirror states data sharing is for
research purposes only. No dataset is redistributed by this repository — only
download scripts.
