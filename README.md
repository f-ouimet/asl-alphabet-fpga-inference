# asl-alphabet-fpga-inference

FPGA and Python scripts for an ASL Alphabet Vision CNN Model

## Description

Fichiers écrits pour le projet thématique du cours MIC5012 à l'Université du Québec à Montréal. Modèle CNN sur FPGA.

## Sources (verilog/sources/)

### memory_streamer (mem_streamer.v, top)

module top de l'accélérateur. Sa machine à états enchaîne conv1 → pool1 → conv2 → pool2 → dense1 → dense2 sur un seul conv_engine et un seul pool_engine, en ping-pong entre ram_a et ram_b. Il génère les adresses d'activations et de poids, et capture les 24 logits. Il contient aussi le MMCM, le reset POR, l'interface SPI et les LED.

### conv_engine

regroupe mac_unit et requantize, et sert aux couches conv et dense. Il accumule un produit act × poids par cycle, puis ramène le résultat en int8 au dernier terme (latence de 4 cycles).

### mac_unit

multiplieur-accumulateur int8 → int32 qui calcule acc = bias + Σ (act − zp_in) × poids.

### requantize

requantification int32 → int8 selon la spec TFLite, en enchaînant srdhm → round_shift → relu_clamp.

### srdhm

multiplication par le multiplicateur Q31 avec arrondi (SaturatingRoundingDoublingHighMul de gemmlowp), en pipeline sur 2 étages.

### round_shift

décalage à droite avec arrondi (RoundingDivideByPOT).

### relu_clamp

ajoute zp_out et sature entre clamp_min et 127. Le ReLU est fusionné en posant clamp_min = zp_out.

### pool_engine

max-pooling 2×2, stride 2. Il garde le max signé des 4 pixels de la fenêtre, sans requantification.
act_ram : RAM d'activations 8 bits, simple dual-port, lecture synchrone (inférée en BRAM). Deux instances : ram_a et ram_b.

### param_rom 
ROM de paramètres générique chargée par $readmemh. Quatre instances : poids, bias, mult et shift.
cnn_params.vh (include, pas un module) : dimensions des couches, offsets dans les ROM et zero-points de quantification.

## Testbenches (verilog/testbenches/)

### tb_mac_unit

vérifie l'accumulation du MAC sur un petit vecteur.

### tb_conv_engine

vérifie deux éléments de convolution avec requantification.

### tb_pool_engine

vérifie deux fenêtres de max-pooling.

### tb_inference

test complet de l'inférence sur l'image de référence, qui compare les 24 logits à golden_logits.mem.
