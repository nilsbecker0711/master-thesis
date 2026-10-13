# Tutor feedback on the thesis draft of 13 October

**Draft:** *Below the Threshold of Sight: Contrast-Sensitivity-Bounded Adversarial Patches for Semantic Segmentation* (main_13-10.pdf, 92 pages)
**Reviewer stance:** fair but harsh. Page numbers are the printed numbers. Section, table and figure numbers follow the draft.
**How this was checked:** I read the whole draft. I recomputed about 70 numbers in the text from the tables. I also checked several method claims against the `patchreach` code in the repository (`patch/csf.py`, `patch/spec.py`, `models/wrapper.py`, `diagnostics/spectral.py`, `losses/adversarial.py`, `scripts/lpips_eval.py`, `lr_sweep.sh`). If a point says *computed*, I ran the calculation myself, and the script is quoted. If it says *inferred*, it is my reading and you should verify it.

---

> **Annotated 2026-10-13.** Section 7 at the end of this file records what was
> recomputed, what was fixed in code, and one place where the review's implied
> conclusion does not survive being carried through (see M3 there).


## 0. Verdict

The draft is well built and honest. The writing is precise. The method is laid out step by step, with a provenance table that separates borrowed parts from new ones (Table 4.2). Every CSF number is read against an unconstrained ceiling and a random control, and that design choice is right. You separate nominal τ from realised visibility, and realised enforcement was a real fix. The limitations sections say things most students hide. Almost every number I checked in the text matches its table.

The weaknesses are in the **evidence** and in **what the visibility model actually does**, not in the prose. If you submitted it today, I would expect an examiner to press hard on the following:

1. **Every quantitative claim rests on one image and one seed.** Your own development log shows nine-point run-to-run swings. Checkpoint selection on the evaluation metric also inflates the small effects.
2. **Most of SETR-PUP's "robustness" to the raw patch comes from the sliding-window geometry, not the architecture.** About 39% of the remote region cannot be reached by a centred patch at 1024 × 2048. I computed this with your own `slide_windows`.
3. **The budget puts about 92% of its power in the diagonal corners of the spectrum, above 0.5 cycles/pixel (up to 54 cpd).** The text, Figure 4.2 and the frequency probe never look there.
4. **The attack is under-optimised in a way that could explain the weak results.** With lr 0.1 and every bin at its bound, nearly every step overshoots the per-bin box and is clamped back, so the optimiser effectively chooses phases only. `lr_sweep.sh` says this itself.
5. **The baseline that would test your premise is missing.** That baseline is a same-footprint ℓ∞/ℓ2-bounded residual, matched on an independent visibility measure. LAP is not that comparison, and the headline "an eighth of LAP's LPIPS" uses the one LPIPS variant most likely to be blind to your residual.

None of these is fatal. Points 2 and 3 are fixes to the text plus cheap analysis. Points 1, 4 and 5 need compute, but in modest amounts. Section 6 below gives a plan in order of cost.

---

## 1. Major issues

### M1. The evidence base: one image, one seed, wrong control, selection bias

**What.** All of Chapter 6 uses image 464 (transfer adds image 42). Each configuration has one run. Section 4.7 (p. 21) says that "four identical single-image runs with a flat learning rate differed by 9.6 mIoU points". You give no run-to-run spread for the cosine schedule that you actually use. Without it, nobody can tell which differences in Tables 6.3, 6.4 and 6.8 are real.

Three further problems make this worse:

- **Best checkpoint selected on the metric you report.** Selecting the strongest checkpoint is fine as an attacker's choice (Section 4.7). But it takes the maximum of a noisy metric over 300 to 600 checkpoints, and that is biased upwards, most of all where the true effect is small. Example: SETR CE at τ = 0.25 is reported as 8.9 points, yet its final patch gives 3.1 (Appendix D.2, p. 69). SETR CE carries † at every τ. For SETR and DeepLabV3+ at small τ, a large share of the "effect" may be selection noise.
- **The CSF arm is compared with the wrong control.** "By more than the random control in every case" (Section 6.4, p. 34) uses the random *raw* patch, which is an occluder of uniform noise. The right control for the CSF arm is the **untrained CSF residual**: the initial random-phase residual at the bound, at each τ, on image 464 at 1024 × 2048, scored with the same best-of-K selection. You only report this control in the 512 × 1024 ablation on another image (Table D.2).
- **Small classes dominate the remote mIoU at small effects.** SETR loses about 9 points with 0.2 to 0.7% of pixels flipped. R101 CE at τ = 0.25 loses 8.5 points with 0.4% flipped. You know this (Section 7.7), yet Section 6.4 and the abstract still lead with mIoU.

**What to do.**
1. Add the untrained-CSF control at each τ on image 464, for all six models. This is cheap: one forward pass per τ.
2. Report final and best side by side in Table 6.3, or select the checkpoint by training loss rather than remote mIoU.
3. Repeat B0, B5 and SETR at τ ∈ {0.5, 1} with CosPGD under three seeds, and give the spread in the text. If you can only afford one extra experiment, this is it.
4. Make the flip rate co-primary with the mIoU drop. For SETR and R101, add per-class IoU changes.
5. Deliver the full-validation means (Section 6.11) for at least raw and CSF at τ ∈ {0.5, 1} with CosPGD on all six models. Fifty images would already change the standing of the thesis.

### M2. SETR-PUP's raw ceiling comes from the sliding windows, not the architecture

**What (computed).** SETR runs with 768 × 768 windows at stride 512 (Section 5.2). On a 1024 × 2048 frame, mmseg's grid (your `slide_windows` in `models/wrapper.py`) gives column windows [0,768), [512,1280), [1024,1792) and [1280,2048). The centred patch covers columns 896 to 1152. It lies only in the second and third windows. Columns 0 to 511 and 1792 to 2047 are scored only by windows that never see the patch.

**Only 61.3% of the remote region is reachable at all.** The raw CosPGD patch flips 60.1% (Table 6.2), so it already saturates the reachable region. At 512 × 1024 a single 768-pixel window row covers both columns of windows and everything is reachable. That is exactly where SETR's raw patch flips 100% (Table D.2). SegFormer's 1024-pixel windows contain the patch in all three windows, so B0 and B5 have full reach. That fits B0's 100%.

Script used:
```python
# windows exactly as patchreach.models.wrapper.slide_windows
SETR 1024x2048: reachable remote fraction 0.613
SETR  512x1024: reachable remote fraction 1.000
SegFormer 1024x2048: reachable remote fraction 1.000
```

**Consequences for the text.**
- Section 6.3, p. 34, reads "Among the transformers, global attention does not imply uniform vulnerability: … SETR-PUP … keeps 40 to 48% of its remote pixels unchanged". The unchanged pixels are mostly the unreachable ones. The sentence has to go.
- Section 7.1 ("Reach is not the limit") and Section 7.7 ("they may lower its ceiling") understate this. The windows do not *may* lower the ceiling; they *set* it, by a computable amount.
- SETR's fractions of its ceiling (16%, Section 6.4) and its transfer ratios use a denominator capped by inference geometry.
- The windows do **not** explain the CSF patch's failure on SETR (0.7% flipped against 61% reachable). Say so, because it sharpens that result.

**What to do.** Report the reachable fraction in Table 5.2. Normalise SETR's flip rates by the reachable region, or show a reach mask. If memory allows, run one whole-image SETR raw attack as a check.

### M3. The budget lives in the spectral corners, which the text and the probe never examine

**What (computed).** The real 2-D DFT grid has radial frequencies up to √2 × 0.5 = **0.707 cycles/pixel, i.e. 54.1 cpd** at your geometry, in the diagonal corners. There Barten's CSF is **1.27**, against 13.4 at 38.3 cpd and 715 at the peak. So the 1/CSF envelope rises about **560 times** from peak to corner, not 53 times.

On the 256 × 256 grid, **91.6% of the power of the normalised envelope lies at radial frequency above 0.5 cycles/pixel, and 99.6% above 0.35**. Those corner bins are only 22% of all bins. This is why the residual in Figure 6.4 is "close to a checkerboard": the (0.5, 0.5) bin *is* a checkerboard. The docstring of `radial_frequency_cpp` ("Max is 0.5 (Nyquist)") is wrong, and the text inherits the error:

- Section 2.2 says the Nyquist is 38.3 cpd, so "the sampling grid cannot represent a frequency that is invisible outright". On the diagonal it represents 54 cpd, at the edge of the visible passband.
- Section 4.2 and Figure 4.2 say the allowance "rises by a factor of about 53". The figure stops at 0.5.
- **The frequency probe's top band ends at 0.5** (`DEFAULT_BANDS` in `diagnostics/spectral.py`, with `keep = f < hi`). The probe therefore never tests the region where more than 90% of the attack's power sits. RQ1 asks which bands the networks read, but it measures bands the attack barely uses.

**Why it matters beyond the text.** A perturbation that is mostly a diagonal pixel checkerboard is (a) probably very fragile to any resampling, JPEG compression, demosaicing or display rendering, and (b) far from anything a physical patch could produce. Both points bear directly on the threat model (M8) and the ethics section.

**What to do.**
1. Fix Sections 2.2 and 4.2 and extend Figure 4.2 to 0.707 cycles/pixel, or plot the 2-D envelope.
2. Add a probe band from 0.5 to 0.71.
3. Plot the radial power spectrum of the saved optimised residuals (`rapsd` already exists in `csf.py`) to confirm they follow the envelope.
4. Run a cheap robustness check: re-evaluate the saved B0 and B5 patches after JPEG q = 95 and q = 90, a 0.5-pixel bilinear shift, and a 0.9× rescale. If the attack collapses, that is a finding. It belongs in Sections 7.6 and 7.8 and strengthens the "diagnostic, not practical" argument.
5. You already implement the SSO CSF with the oblique effect. It would lower sensitivity on the diagonal further, so say whether the isotropic model is conservative here.

### M4. Properties of the visibility model that the thesis does not state

Section 7.5 discusses three approximations. These four are missing, and an examiner who knows psychophysics will find them:

- **Pooling over DFT bins makes τ depend on patch size and bandwidth (computed).** Normalisation divides by v_full = (3N)^{1/3}. That gives **46.3 for a 256 patch and 29.2 for a 128 patch**, so every bin sits at 0.022 and 0.034 of threshold respectively. Pixel RMS with all bins at the bound and τ = 1 is **0.137 against 0.109**. At the same τ, the bigger patch therefore carries **26% more RMS**. A human seeing equal-RMS noise over a larger area finds it *easier* to detect, not harder (spatial summation), so the dependence runs the wrong way. The same mechanism favours wide bands in the probe: at equal V, RMS grows as N^{1/6}. Real channels sum energy within a channel, which is closer to β = 2, and then pool across channels with β ≈ 3 to 4. Pooling bins with β = 3 under-counts broadband noise by roughly n^{1/6} per channel.
- **No spatial localisation.** The Fourier magnitudes discard where in the patch the energy sits. Your own "local CSF visibility" is 0.34 at τ = 0.25 and up to **2.2 at τ = 1** (Table D.16), against a pooled τ. Its definition is still a TODO (Section 4.11). Either define it and discuss it as a second visibility number, or drop it.
- **Colour.** An achromatic CSF is applied separately to R, G and B, and the results are pooled. Luminance is a weighted sum of the channels, and chromatic sensitivity falls off much earlier. The per-channel treatment is a choice that needs one sentence of justification, or an opponent-space version (luminance plus two chromatic channels, as ColorVideoVDP or castleCSF do).
- **Your own luminance analysis undermines the headline numbers.** Section 7.5 says the mid-grey budget is **1.9 to 3 times too permissive** on Cityscapes. Then "B0 reaches its ceiling at half the nominal threshold" (abstract) is roughly one JND after correction. "Most of its effect at a realised visibility of about 1 JND or less, nominally at or below threshold" (Section 6.4) becomes about 2 to 3 JND. You state the correction and then never apply it to a claim. Apply it in the abstract, Section 6.4 and Section 8.1.

**What to do.** You do not need to change the attack. Add one independent observer model that filters by the CSF and pools over **space**, for example the ModelFest standard observer (Watson and Ahumada 2005, *JoV*), which uses your SSO CSF. Report every patch under it. Because any linear-filter plus Minkowski model is 1-homogeneous, your existing `fit_to_visibility` rescale could also *enforce* it with no other change. That would be a strong ablation.

### M5. The attack is probably under-optimised, and τ is not the single knob claimed

**What.**
- **The per-bin box is a much smaller set than V_β ≤ τ.** `normalise_budget_to_tau` turns the pooled bound into a box: every bin may use only its 1/(3N)^{1/3} share. The optimiser cannot move energy to bands the network reads. Combined with the initialisation (every bin at its bound, phases from `randn*10`), it can mostly only rotate phases. This contradicts the framing "τ, one parameter, bounds how visible the patch may be". τ bounds a fixed-shape box, and the attacker is weaker than τ permits.
- **The learning rate is far too large for this parameterisation.** `lr_sweep.sh` documents it: under the PGD parameterisation "param rms 0.0272 → an Adam step of 0.2 is 7.4× it … the scale-equivalent … ~= 0.0055". The main runs use 0.1, about 3.7 times the parameter RMS per step. Nearly every bin is clamped back to its bound after every step, until the cosine schedule decays. That fits 18 of 60 runs ending weaker than their best checkpoint (Section 6.4), and Section 4.7's remark that a flat schedule makes the result "wherever the walk happens to stop".
- **"The same lr for every arm, so that a difference between arms is not a difference in tuning" (Section 5.4) is false in effect.** The raw arm optimises sigmoid logits of order 1. The CSF arm optimises a pixel residual of RMS about 0.03. The same nominal lr is a 30× different step.

**Why it matters.** "SETR-PUP is close to immune" and "the budget is not what stops the attack" are headline claims. They are not safe until an lr sweep on the CSF arm rules out an optimisation artefact.

**What to do.**
1. Run `lr_sweep.sh` on the CSF arm for B5 and SETR at τ = 0.5 and τ = 1, with lr ∈ {0.003, 0.01, 0.03, 0.1}.
2. Add a "pooled-only" variant: drop the per-bin projection and keep the realised-visibility rescale. That is almost a one-line change, and it measures the threat at a given τ honestly.
3. Log the fraction of bins at their bound over training.
4. If you cannot run these, say plainly in Sections 6.4 and 7.7 that SETR's result may be an optimisation artefact, and remove "immune".

### M6. The baseline that tests the premise is missing

The thesis implies that shaping the residual by the CSF buys more damage per unit of *independently measured* visibility than a standard bound. Nothing in Chapter 6 tests that.

LAP (Section 6.9) aims at a different goal (looking like a natural object), runs on a 128 grid and on silhouettes, needed recalibration, and is matched at a single point (45.5 against 46.9). You say yourself that it is "not a verdict on LAP".

**What to do.** Run a same-footprint residual under ℓ∞ PGD at ε ∈ {2, 4, 8, 16}/255, and ideally an ℓ2 version or a flat-spectrum envelope under the same realised rescale. Use the same models, CosPGD and image. Plot remote drop against PSNR, VGG-LPIPS and ColorVideoVDP JOD for both families. If the CSF curve dominates, you have the result your title promises. If it does not, you have an honest and still publishable negative. The ℓ∞ projection is trivial to add to `spec.py`.

### M7. The perceptual evaluation is read more favourably than it supports

- **The AlexNet LPIPS headline.** AlexNet's first layer is an 11 × 11 convolution with stride 4: the same strided-stem argument you make about networks in Section 1.2. A pixel-scale checkerboard is exactly what it is likely to miss. Your own tables are consistent with that. The CSF residual at τ ≥ 1 has RMS 0.041 to 0.062 (Table D.16), two to three times the RMS of ±8/255 uniform noise (0.018). Yet its full-image AlexNet LPIPS (≤ 0.004) is *lower* than that noise's (0.006). VGG (stride-1, 3 × 3) puts the CSF patch at 0.39 against LAP's 0.57, a ratio of 0.7, not one eighth. The abstract's "an eighth as far … in LPIPS" and Section 8.1's "eight times its LPIPS" should report both variants, or lead with VGG.
- **A missing reference point.** Score the *untrained* CSF residual (same envelope, random phase) with the whole battery. That separates "the CSF shape is cheap under metric X" from "the optimiser kept it cheap".
- **The no-reference metrics are out of their regime.** BRISQUE rates the CSF crop as *worse than random pixels* (about 90 against 43). A metric that finds uniform noise more natural than your residual is not measuring naturalness here; it reacts to the high-frequency statistics it was built on. "Conspicuously unnatural" (Section 6.8) overreads it. Also, "rate the patched crops as degraded at least as much as random pixels" (Sections 7.3 and 8.1) is false for NIQE at τ = 0.25 (22.4 and 21.2 against 31.9).
- **ColorVideoVDP is not independent of the budget's geometry.** The thesis leaves its settings as a TODO (Section 5.6), but `scripts/lpips_eval.py` answers it. `ppd_from_cfg` derives pixels per degree from the run's own `csf_pixel_size_cm` and `csf_viewing_distance_cm` (about 76.6 ppd), and the display is `vvdp_display_photo_eotf(Y_peak=100)`. The docstring says this was deliberate: "CVVDP and the tau budget describe one observer rather than two". That is a defensible design, but then Sections 2.6 and 6.8 cannot say the measure "shares none of the budget's assumptions" or is "independent of it". It shares the viewing geometry, which is the assumption with the largest effect (Section 2.2: about 5× on a laptop). Say so, and add one ColorVideoVDP column at a second geometry, such as the 15.6-inch laptop at 50 cm from Section 2.2. Separately, the noise reference for JOD is computed on the full image (Table D.13) while the CSF JOD is on the crop (Table 6.9), so those two cannot be compared.
- **Claims from downsampled figures.** Figure 6.2 says the τ = 1 patch "cannot be seen in the image at this scale". Section 6.4 says "In the image, only the unconstrained patch can be seen". Printing a 2048-pixel frame at page width low-pass filters away exactly the frequencies the residual occupies, so the statement is true by construction and says nothing. Delete both sentences. Point readers to full-resolution crops in supplementary material, with your viewing instruction from Figure 6.4.

### M8. The threat model and the title

The motivation (Section 1.1) is physical: stickers, posters, road patches. The attack is digital, per-image, scene-specific and, by M3, essentially a pixel checkerboard. Section 7.2 concedes that "the practical threat … is the per-image, digital one: an attacker who controls the image pipeline". Two questions follow, and an examiner will ask both:

1. **If the attacker controls the digital image, why restrict to a patch?** Answer it: locality as a stealth or forensic constraint, or comparability with the patch literature. Ideally add the full-image CSF-bounded perturbation as a reference point, since that is the natural home of a CSF bound (as in Deng and Karam).
2. **Who is the observer?** At 0.0131°/pixel, it is someone viewing the frame at about 223 PPI from 50 cm. Name the scenario, for example a human auditor reviewing recorded frames on a monitor, in Sections 1.1 and 4.1.

**The title, "Below the Threshold of Sight", is not supported** by a thesis that runs no human study, admits a 1.9 to 3× luminance bias, reports realised visibilities up to 1.5 and local visibilities up to 2.2, and calls τ "a relative axis". The scope statement (Section 1.5) is exemplary. The title, the heading of Section 7.3 ("Invisible but detectable") and phrases such as "a perturbation that no person sees" (Section 2.7) contradict it. Make them consistent. A title such as "Contrast-Sensitivity-Bounded Adversarial Patches for Semantic Segmentation" loses nothing.

### M9. Novelty claim and related work

- **"Inverted CSF as a hard budget: no external source" (Table 4.2, Section 4.12) is overclaimed.** Per-frequency amplitude budgets derived from visibility thresholds are standard in perceptual watermarking (e.g. Podilchuk and Zeng 1998, *IEEE JSAC*; Watson's DCTune, 1993). In adversarial work there is Deng and Karam 2020, which you cite and which uses JND thresholds per frequency as perturbation bounds. Narrow the claim to "for adversarial patches on semantic segmentation", and add the watermarking lineage to Chapter 3. Also discuss frequency-constrained imperceptible attacks: Luo et al. 2022 (CVPR, SSAH) and Duan et al. 2021 (ICCV, AdvDrop).
- **Patch defences built on high-frequency or gradient statistics** belong in Section 7.3, which predicts that such detectors would work: Local Gradient Smoothing (Naseer et al. 2019, WACV) and Hayes 2018 (CVPRW). LGS is cheap enough to actually run on your saved patches.
- **Yuan et al. (Section 3.3, p. 11).** As written, the numbers contradict the sentence. "Vulnerability grows with reach", yet mIoU under attack *rises* from 5.86 (convolutional) to 74.57 (global attention), which reads as *less* vulnerable. Either the metric is something else (a targeted success mIoU?) or the sentence is wrong. Resolve the TODO before quoting anything.
- **Table 3.1** marks this thesis "Universal? both". The universal results do not exist yet.

### M10. Pending results and scope

RQ3 (universality) and the full-validation means are TBD in the abstract, in Sections 5.10, 6.6, 6.11, 7.2, 8.1 and 8.2, and in Table 6.7. Decide now what you can deliver at a quality you can defend.

My recommendation: **the full-validation means come before the universal patch.** They underpin every other claim, and the universal patch adds a new experiment with its own tuning problems. If the universal patch slips, demote RQ3 to the predictions that already exist in Section 6.6. Soften "lower bound": a per-image patch evaluated on a new image does not bound what universal training can reach. Present it as a heuristic expectation.

Six RQs is also a lot. RQ5 (CosPGD against CE) is an ablation, not a research question. RQ1 is inconclusive by its own pre-set criterion. A tighter set of three or four questions would read better. The RQ order also does not match the results order: Section 6.5 answers RQ4 before Section 6.6 answers RQ3.

---

## 2. Moderate issues

**RQ1, the frequency probe (Sections 5.7, 6.2 and 8.1)**
- You set the 0.5% threshold in advance and the verdict is "inconclusive for all six". Contribution 2 and the RQ1 answer still present the band profile as a finding. Lead with "inconclusive" and present the profile as exploratory.
- "The most sensitive band is 0.20–0.35 for five of the six models" is wrong. SETR's mean peak is band 6 (0.43 against 0.35), as the next paragraph says, so the count is **four of six**.
- R101's mean peak (band 4) is driven by one image (SD 0.63). Its per-image peaks are band 5 in four of five images. Report medians.
- Band 6 reached only 2.3 JND, so it is not at equal cost, and comparisons that involve it (SETR) are not like for like.
- A stronger instrument for "which bands does the network read": the radial spectrum of the input gradient of the loss inside the patch, or band-restricted optimised attacks. Both are cheap given your code.

**The stride argument (Sections 1.2, 2.1 and 6.2).** "The Nyquist frequency of a stride-4 embedding is 0.125 cycles per pixel" treats a strided multi-channel convolution as a sub-sampler. It is not one: it can alias high frequencies into channels without discarding them, as you say yourself about SETR in Section 5.2. The premise that you then refute in Section 6.2 is a straw man. Reframe it.

**A cheap test of your SETR hypothesis (Section 7.1).** Compute the singular values of SETR's 768 → 1024 patch embedding, and project the embedding onto the 16 × 16 DCT basis. If high-frequency DCT components map to small gains, you have evidence for "the token changes too little". The same for the first-layer frequency responses of B0 and B5. This is hours of work, not days, and it turns the hypothesis in Section 7.1 into a finding.

**B0 against B5 stems (Sections 5.2 and 7.1).** "3 to 64 channels at stride 4 … expansive" only makes sense if you count 4 × 4 × 3 = 48 new inputs per output position. As written it reads as wrong, because a 7 × 7 × 3 kernel sees 147 inputs. Explain the count.

**Transfer (Section 6.5).**
- Is the range fit and the realised-visibility rescale recomputed on image 42? If so, report the realised visibility and RMS of each transferred residual. Otherwise "scene dependence" is confounded with "the residual got scaled down on new content".
- Add a random raw control on image 42. You already note its absence, but it is cheap.
- One target image is thin. Use five.
- R101 CosPGD retains 115% on image 42. One sentence on why.

**RQ5 (Section 6.7).** The "hypothesis of Section 1.3" is never stated in Section 1.3; RQ5 is only a question. Differences under 6 points sit inside the 9.6-point run noise of M1. Your loss is also not CosPGD: the cosine is detached and the optimiser is Adam, not sign steps. Call it "cosine-weighted CE (after CosPGD)" throughout.

**LAP (Sections 5.8 and 6.9).**
- "With either published weight setting, LAP lowers the remote mIoU by 0.3 to 2.6 points, no more than the random patch (0.4)" is false: 1.3 and 2.6 exceed 0.4.
- "α 220 to 420 times smaller than published" holds only against the "real" setting (0.1). Against the "train" setting (1e-4), the calibrated α (2.39e-4 and 4.53e-4) is 2.4 to 4.5 times *larger*. The same applies in Section 7.4.
- The ‡ rows of Table 6.11 take the attack columns from the final patch and the perceptual columns from the saved best. "At equal strength" then pairs numbers from different patches. Fix the checkpoint mismatch at its root. The same bug affects B5 CE at τ = 1. A reader will wonder what else it touched, so explain the cause in one sentence.
- A single calibrated setting matched at one point is weak. Sweep α and compare frontiers.

**Dead and unstated parameters.**
- **A_max = 0.25 never binds (computed).** The largest pre-normalisation allowance is 0.197, and after normalisation it is of order 1e-3. Remove the parameter, or say that it is inactive.
- **X₀ = 60°** (Section 2.3) is the object size in Barten's area term. Your stimulus is 256 × 0.0131 ≈ 3.4°. Justify the choice (it is conservative at low frequencies, I believe, but say so).
- **The pupil formula** d = 5 − 3 tanh(0.4 log₁₀ L) is attributed to Le Grand. My understanding is that Le Grand's formula has different constants and that this form is Barten's own. Check it against Watson and Yellott (2012, *JoV*).
- **Initial phases (TODO in Section 4.7).** The code answers it: `randn(3,S,S)*10`, seed 42, projected onto the bound, so the phases are those of i.i.d. Gaussian noise (uniform).
- **The realised-visibility scale is detached** in `fit_to_visibility`. State that in Section 4.5.
- **The random control and the raw arm start from different patches.** The raw arm starts mid-grey (`raw_init='grey'`, Table C.1); the random control uses uniform random pixels. Say so in Section 4.9 so nobody reads the control as the raw arm's step 0.

**The attacker uses ground-truth labels**, both in the loss and in checkpoint selection. That is standard for white-box evaluation, but state it. A variant with the clean prediction as pseudo-label would be more realistic.

**Reach analysis.** The thesis builds on patch reach (Section 3.3) but never shows flip rate as a function of distance from the patch. One figure per model would be cheap and very informative, especially next to M2.

---

## 3. Factual and consistency errors found while checking numbers

| Location | Problem |
|---|---|
| Abstract | "reaches about half of it on SegFormer-B5 and DeepLabV3+". The DeepLabV3+ fractions are 32%, 78% and 56% (Section 6.4). Say "a third to three quarters". |
| Abstract and Section 8.1 RQ6 | "an eighth … in LPIPS" uses AlexNet only; VGG gives 0.7× (M7). |
| Section 2.2 | "cannot represent a frequency that is invisible outright": the diagonal reaches 54 cpd (M3). |
| Section 4.2 and Figure 4.2 | "factor of about 53": about 560 to the corner (M3). |
| Section 6.2 | "five of the six models" peak in 0.20–0.35: four (SETR peaks in band 6). |
| Section 6.3 | "SETR-PUP … keeps 40 to 48% of its remote pixels unchanged" is explained by the windows (M2). |
| Section 6.9 | "no more than the random patch (0.4)": 1.3 and 2.6 are more. |
| Sections 6.9 and 7.4 | "α 220 to 420 times smaller than published": only against one of the two published settings. |
| Sections 7.3 and 8.1 | "at least as much as random pixels": not for NIQE at τ = 0.25. |
| Tables 6.2/6.3 against Table 6.6/D.5 | B5 CE raw white-box is 32.9 in one place and 33.4 in the other. Explained by the re-evaluation in Section 5.5, but footnote it where both appear. |
| Section 6.4 | "In the image, only the unconstrained patch can be seen": unfalsifiable at print scale (M7). |
| Section 3.3 | The Yuan et al. numbers contradict the sentence they support (M9). |
| `radial_frequency_cpp` docstring | "Max is 0.5": it is 0.707 (code, not thesis, but it caused the text error). |

Everything else I recomputed matches: the fractions of the ceiling, the counts of † runs (18 of 60), all of Table 6.8, the transfer means, the LPIPS ratios, the probe peak counts (22/5/3), the geometry conversions (79 cm at 141 PPI, 44.7 cpd for the road patch, the five-times laptop factor) and the κ = 4 derivation.

---

## 4. Writing, structure and presentation

**What works.** The prose is clean, the paragraph headings carry the argument, and Chapter 4 follows the computation in order. Figure 4.1, the provenance table and Algorithm 4.1 make the method reproducible. Figures 6.1 and 6.4 are the best in the thesis. The limitations are stated more honestly than in most published papers.

**What needs work.**
- **Claims are calibrated unevenly.** Sections 1.5, 4.8 and 7.5 hedge carefully, while the abstract, the title, the heading of Section 7.3 and individual sentences in Chapter 6 overclaim. An examiner will quote the scope section back at you. Do one pass where every sentence that contains "invisible", "cannot be seen", "below threshold" or "immune" is checked against Section 1.5.
- **Development anecdotes in the main text** ("four identical runs differed by 9.6", "a residual … measured at 5.0 after clipping", "±7.65") are persuasive but unsupported. Either show them as a small table in the appendix or cut them. Section 7.4's list of "fixes found during development" is honest, but it undercuts the argument that the CSF bound is "set once" in contrast to LAP. q, m_c, β, κ, A_max and the geometry are tuned constants too. Reframe that argument more modestly.
- **Float placement.** Tables 6.10 and 6.11, which Sections 6.8 and 6.9 discuss, appear after the Section 6.11 heading. Page 76 is blank apart from the heading "D.4 Perception tables". Section D.3's text sits on p. 70 with its tables from p. 72. Use `\FloatBarrier` at section ends and `[p]` floats in the appendix.
- **Figure 6.2.** The prediction overlay is too small and too dark to read: the car class in dark blue on a dark image. Show the predictions without the image, at a larger size, or crop to a region.
- **The Section 2.7 footnote** cites a blog post. That is acceptable, but use the paper (Falck et al. 2025, already cited) as the primary source.
- **The declaration (p. 84).** Strike the alternatives so it reads "Masterarbeit". Fill in the "Useful?" column. More importantly, the AI-use table says Claude drafted text "throughout from the author's outline and notes". Confirm with your supervisor, *before submission*, that this level of AI drafting is acceptable under your examination regulations and is declared in the form they expect. This is the one item in this review with consequences beyond the grade.
- **Other open placeholders:** matriculation number, date, supervisor, the two TBD sentences in the abstract and Section 8.2, the TODOs in Sections 3.3, 3.5, 4.7, 4.11, 5.1, 5.4, 5.6, Table 5.3, Appendix C and Section 7.8. Several can be closed from the code today: initial phases (Section 4.7), the IQA-PyTorch version (`pip show pyiqa`), and the ColorVideoVDP display (the run's own geometry, about 76.6 ppd, and Y_peak = 100 cd/m² in `scripts/lpips_eval.py`; see M7).

---

## 5. Questions to expect at the defence

1. Why should I believe a ranking of six architectures measured on one image with one seed, when identical runs differ by nine points?
2. SETR's raw patch leaves 40% of the frame untouched. Is that the attention mechanism or the sliding window? *(M2: have the reachable-fraction number ready.)*
3. Your residual is a checkerboard. What happens to the attack after JPEG compression or a half-pixel resize?
4. Does the CSF shape buy anything over an ℓ∞ bound at the same PSNR or JOD?
5. Your lr is about 30 times larger relative to the parameter than in the raw arm. How do you know SETR is not simply under-attacked?
6. Pooling over Fourier bins makes your budget depend on patch size. Why is that a reasonable observer model?
7. With your own luminance correction, is τ = 0.5 still below threshold?
8. Who is the human observer in your threat model, and why would the attacker restrict themselves to a patch?
9. What exactly is new compared with perceptual watermarking and Deng and Karam?

---

## 6. Plan, in order of cost

**Tier 0: text and analysis only, no GPU (one to two days)**
- Rewrite the SETR reach passages (M2) and add the reachable fraction to Table 5.2.
- Fix the Nyquist and corner statements and Figure 4.2 (M3).
- Apply the luminance factor to headline claims, and align the title, abstract and Section 7.3 with Section 1.5 (M4, M8).
- Narrow the novelty claim and add the missing related work (M9).
- Correct every item in the table in Section 3.
- Close the TODOs that the code answers. Settle the AI declaration with your supervisor.

**Tier 1: cheap compute (under a GPU-day in total)**
- The untrained-CSF control at each τ on image 464 (M1).
- JPEG and resampling robustness of the saved patches (M3).
- Add the 0.5–0.71 probe band (M3).
- SVD of the SETR embedding and the frequency response of the B0/B5 stems (Section 2).
- Realised visibility of transferred residuals, a random raw control and four more target images (Section 2).
- Score the untrained CSF residual with the perception battery, and add ColorVideoVDP at a second viewing geometry (M7).
- Radial spectra of the optimised residuals (M3).

**Tier 2: the core (a few days)**
- Three seeds for B0, B5 and SETR at τ ∈ {0.5, 1} (M1).
- An lr sweep on the CSF arm for B5 and SETR (M5).
- The same-footprint ℓ∞ baseline with drop-against-visibility curves (M6).
- Full-validation means on at least 50 images for raw and CSF at τ ∈ {0.5, 1} with CosPGD (M1, M10).

**Tier 3: if time remains**
- Universal patches (RQ3).
- The pooled-only variant (M5).
- A spatial-domain observer model as a second enforcement (M4).
- LGS as a detector (M9).

If you do Tier 0 and Tier 1 and the first two items of Tier 2, the thesis moves from "careful engineering on one image" to "a defensible empirical study". Tier 0 alone already removes most of the ammunition an examiner currently has.


---

## 7. Annotations, 2026-10-13

Worked through against the code. **[checked]** = recomputed here, with the
source named. **[new]** = not in the review. **[done]** = committed on
`claude/thesis-feedback-review-48a395`. **[open]** = untouched.

Headline: every claim the review marks *computed* reproduces, to the digit. Two
code changes came out of it. One of the review's implied conclusions does not
survive -- the entry under M3 below is the most important in this section.

### What was verified

| Claim | Review | Recomputed |
|---|---|---|
| power of the normalised envelope above 0.5 c/px | 91.6% | **0.9165** |
| above 0.35 c/px | 99.6% | **0.9960** |
| bin fraction above 0.5 c/px | 22% | **0.2174** |
| radial maximum of the rfft grid | 0.707 | **0.70711** |
| corner frequency / CSF there | 54.1 cpd, 1.27 | **54.13 cpd, 1.2722** |
| axis-Nyquist CSF / peak CSF | 13.4 / 715 | **13.399 / 714.91 at 2.34 cpd** |
| peak-to-corner budget ratio | ~560x | **562x** |
| v_full at size 256 / 128 | 46.3 / 29.2 | **46.269 / 29.218** |
| RMS at all-bins-at-bound, tau 1 | 0.137 / 0.109 | **0.1370 / 0.1086 (+26%)** |
| `A_max` = 0.25 never binds | max 0.197 | **0.19652, at both sizes** |
| SETR reachable remote fraction at 1024x2048 | 0.613 | **0.6129** |
| `fit_to_range` can only shrink | stated | **`q.clamp(0.0, 1.0)`** |

`DEFAULT_BANDS` did stop at 0.50 and `radial_frequency_cpp`'s docstring did say
"Max is 0.5". M2, M3 and the computed half of M4 are facts.

### M1 -- evidence base

**[new, ordering]** Item 5 (full-validation means) should come *after* the lr
sweep, not before. `lr_sweep.sh` exists -- 7 lrs x 1000 steps on the CSF arm,
4 archs, `analysis/pick_lr.py` as selector -- and nothing in
`docs/experiment_log.md` records it running. Spending the plan's largest
compute block on a possibly mis-tuned attack means re-running all of it.
**[open]**

### M2 -- SETR reach

**[checked]** The grid reproduces from `slide_windows`: columns [0,768),
[512,1280), [1024,1792), [1280,2048). The centred 256px patch spans columns
896-1152, lies only in the middle two, whose union is [512,1792). Reachable
remote fraction = (1024*1280 - 256^2) / (1024*2048 - 256^2) = **0.6129**.

**[new]** Two cautions. If flip rate is normalised by the reachable region, the
remote mIoU must be recomputed on the same region or one table carries two
denominators. And the windows are not an artefact to be discounted:
sliding-window inference is the deployed configuration, so the honest sentence
is "inference geometry, not architecture", not "the number is wrong".
**[open]** -- text and Table 5.2.

### M3 -- the spectral corners

**[new]** There were **three** copies of the 0.5 ceiling, not one: the probe's
`DEFAULT_BANDS`, `csf.report()`'s own hardcoded band list -- so the budget
table that exists to sanity-check the premise could not have revealed the gap
either -- and `--n_bands`' `linspace(0.0, 0.5, n+1)`.

**[done]** One list now, `csf.RADIAL_BANDS`, with `(0.50, 0.71)` added.
`ViewingGeometry.describe()` printed only the per-axis Nyquist with the gloss
"the grid does not reach the human cutoff"; it now prints both figures, since
the diagonal reaches 54.1 cpd *inside* the passband. The budget table at size
256, tau 1 now reads:

| band (c/px) | CSF | budget |
|---|---|---|
| 0.20-0.35 | 98.93 | 0.0028 |
| 0.35-0.50 | 28.31 | 0.0103 |
| **0.50-0.71** | **7.75** | **0.0429** |

**[checked]** Item 5: isotropic Barten is the conservative direction. The
oblique effect makes real diagonal sensitivity *lower* than isotropic says, so
the isotropic model grants *less* amplitude at the corner than SSO would. Worth
stating -- it favours the thesis, but it also means the corner concentration
would be more extreme under SSO.

**[new] THE CORRECTION, and it cuts against item 4.** "Mostly a pixel
checkerboard, so probably very fragile to resampling" compares surviving
*fractions*. A network sees surviving *amplitude*. Both arms at tau 1, all bins
at bound, RMS after the filter:

| filter | 1/CSF envelope | band-cut at 0.35 |
|---|---|---|
| none | 0.1370 | 0.0119 |
| 2x2 box (= half-pixel bilinear shift) | **0.0119** | 0.0072 |
| 4x4 box | **0.0050** | 0.0020 |
| Gaussian sigma 0.5 px | **0.0257** | 0.0076 |
| Gaussian sigma 1.0 px | 0.0021 | **0.0026** |

The corner's amplitude-per-JND advantage largely survives the attenuation:

| band (c/px) | cpd | CSF median | amplitude per JND |
|---|---|---|---|
| 0.02-0.05 | 1.6-3.8 | 698 | 0.00072 |
| 0.10-0.20 | 7.7-15.3 | 255 | 0.00196 |
| 0.20-0.35 | 15.3-26.8 | 91 | 0.00550 |
| 0.35-0.50 | 26.8-38.3 | 26 | 0.01929 |
| **0.50-0.71** | 38.3-54.1 | 7.8 | **0.06429** |

A 12x exchange rate against roughly 7x in surviving amplitude, so the crossover
sits near a 1-pixel-sigma blur. The corner spending may well be the right bet,
and the open question is empirical: what do the networks do above 0.5 c/px?
That is what the new probe band measures -- which is why the probe should run
before any band-limited attack arm.

Item 4's fragility prediction does still hold in its own terms: the control arm
loses 91% of its amplitude (0.1370 -> 0.0119) to a half-pixel shift.
**[open]** -- cheap to confirm on the saved patches.

### M4 -- the visibility model

**[checked]** The pooling arithmetic reproduces: v_full 46.27 at size 256 and
29.22 at 128, RMS 0.1370 against 0.1086, i.e. the larger patch carries 26% more
RMS at the same tau, and the dependence runs against spatial summation as
stated.

**[new] direct evidence for the "no spatial localisation" point.** At tau 1
with `csf_enforce=realised`, the realised residual's **absolute maximum is
0.528** -- a pixel moving half the dynamic range while the pooled index reports
0.966 JND. The band-limited arm at the same nominal tau has absmax 0.054. The
pooled Fourier index cannot see this; it is the cleanest single number for this
argument and belongs in Section 7.5.

**[checked]** `fit_to_range` clamps its scale to <= 1, so any 1-homogeneous
bound -- an l_inf epsilon included -- survives it. That is what makes both the
spatial-observer enforcement ablation and the M6 l_inf arm cheap.

### M5 -- under-optimisation

**[checked]** The `lr_sweep.sh` quotes are accurate and the script exists. Item
1 is written but unrun.

**[new]** Item 3 is already done and the answer is free: `scripts/overfit.py`
writes `final_frac_at_bound` and `final_spend_mean`,
`best_checkpoint_stats.py` writes `best_frac_at_bound`, and
`analysis/build_index.py` carries both. Read them off the existing CSVs before
spending a GPU-day.

**[new]** The mechanism claim is weaker than stated. All bins at bound at tau 1
is RMS 0.1370, but the reported CSF residual RMS at tau >= 1 is 0.041-0.062 --
about a third. The optimiser did move bins substantially *down*, so "it can
mostly only rotate phases" does not follow. The CSV columns settle it.

**[new]** M5 is a **prerequisite** for M6, not a parallel task. An l_inf arm run
at lr 0.1 against a box of radius 0.03 reproduces exactly the pathology M5
alleges, and the comparison is then void. One job, one step rule expressed as a
fraction of the constraint radius.

### M6 -- the missing baseline

**[new]** The three proposed axes will disagree and the review does not say
which arbitrates. PSNR is the axis CSF shaping is *designed* to lose on.
ColorVideoVDP takes its ppd from the run's own `csf_pixel_size_cm` /
`csf_viewing_distance_cm` and uses a CSF internally, so it is partly circular --
M7 says so itself. That leaves VGG-LPIPS as the one independent axis. Name the
arbiter in advance; if the title is to claim anything about sight, a small
forced-choice study is the real one.

**[new]** Make the **flat envelope** the primary control and l_inf the
secondary. l_inf changes four things at once -- constraint geometry, domain,
no Minkowski pooling, no realised rescale -- so it cannot attribute a
difference to the CSF *shape*. Keeping every other term fixed and replacing
`B(f) ~ 1/CSF` with `B(f) = const` isolates exactly the thesis's claim, and is
now a one-line change on top of `csf.band_limit`.

**[new]** The l_inf ladder is also the missing middle of a figure that already
exists: the raw arm is the epsilon -> infinity endpoint of that family, so the
two existing arms and the baseline are one continuous curve.

**[done, partial]** `--csf_max_cpp` is the *internal* control (does the shape
matter). The *external* comparison (does CSF shaping beat a standard bound)
remains **[open]**.

### M7 -- perceptual evaluation

**[checked]** `scripts/lpips_eval.py` does derive ppd from the run's own
geometry, so "shares none of the budget's assumptions" is not defensible.
**[open]** -- lead with VGG, add the second viewing geometry, score the
untrained residual, drop the naturalness reading of BRISQUE/NIQE.

### M8, M9, M10

**[open]**, all text-only except LGS. Note `defences/` already has the
registry: `build_defence("lgs")` raises as an unimplemented name while JEDI and
SAC are implemented, so adding LGS as a detector is small.

### Committed in this session

| Commit | What |
|---|---|
| `1371bd1` | `--csf_max_cpp`, a high cut applied *before* the tau normalisation, so live bins absorb the allowance. Off by default. |
| `3a237a6` | `csf.RADIAL_BANDS` with the 0.50-0.71 band, shared by the probe and `csf.report()`; both Nyquist figures in the geometry log; `--n_bands` ceiling raised. |
| `c923c8e` | Reverted the launcher edits; the exact commands are in that commit's message. |

What the band cut does, measured at size 256, tau 1:

| | no cut | `--csf_max_cpp 0.35` |
|---|---|---|
| realised visibility (tau 1 requested) | 0.966 | **1.000** |
| residual RMS | 0.1303 | **0.0119** |
| residual abs max | 0.528 | 0.054 |
| RMS inside 0.20-0.35 c/px | 0.0140 | **0.0201** |
| live bins | 33,018 | 12,700 |

The amplitude scale-up on the surviving bins is only 1.38x, because
`normalise_budget_to_tau` counts bins rather than power. The cut arm is
*more* visible by the pooled index, so the comparison is not tilted in its
favour.

### Back-compatibility

Checked by running both trees side by side rather than by inspection: a git
worktree at the merge base, the same dump script in each, 505 observables
compared bit for bit (every `param`, `_csf_budget`, `_csf_values`, `render`,
`residual`, `apply`, `footprint` and every `stats()` entry, across mode='csf'
at two sizes x pgd/squash x nominal/realised x two taus, the barten and sso
models, three `min_cycles` values, a shaped squash patch, universal_csf across
lref x clip/fit x nominal/realised, three raw variants, and five Adam+project
training steps per csf configuration).

**497 identical. 8 differ, all of them the logged `frac_at_bound` on
universal_csf only**, by 0.011 at size 32 -- the denominator moved from all
bins to live bins, matching `_spectral_occupancy`. mode='csf' is unaffected
because it already reported over live bins. Nothing consumes that number but
the result CSVs. Every training trajectory is bit-identical.

A checkpoint written by the old code and loaded by the new one: all 10 tensors
and stats identical, the config differing only by the new `csf_max_cpp: 0.0`
field.

Unrelated: `tests/test_core.py::test_tv_zero_on_constant` fails on a clean
checkout of `main` too (tv_loss returns 0.000675 for a constant image).

### Next, in order

1. `measure_frequency_sensitivity.py` with the seven bands, native resolution,
   b0/b5/setr, the tau ladder, plus `--region full` and `--normalise rms`.
   The number that matters is the top band's flip rate per JND. A
   `range-limited` flag on that band is also a result: amplitude per JND is
   0.064 there, so it may not reach tau on real content without clipping, which
   is where the realised-vs-nominal gap comes from.
2. The `final_frac_at_bound` / `final_spend_mean` columns in the existing CSVs
   (free, and the first read on M5).
3. The lr sweep (M5), before the full-validation means.
4. The flat-envelope and l_inf arms (M6) under one shared step rule.
5. The band-cut arm, once step 1 says which way it should go.
