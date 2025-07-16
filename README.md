# Buggi-2
Buggi-2, named after the "Kalki: 2898 AD" movie's Buggi character, is a language model trained on the Telugu language (CulturaX dataset on HF) using DGX (version 1 with relatively smaller data), and using A6000 (version 2 on CulturaX). Will host it on GCP if I get to use GCP's 300$ credit with GPU for some time.

### Sample results
#### 98M, Half of CulturaX `te` data:
```
Sample 1/3:
--------------------------------------------------
Generated: జగన్ మోహన్ రెడ్డి, మాలకొండయ్య, కేవీ రమణారెడ్డి, శ్రీనివాస్, నాగశేఖర్, రఘునాధ్ గౌడ్, కృష్ణ, ఆంజనేయులు, శ్రీనివాస్ గౌడ్, శేషయ్య, పాషా, శోభ, శ్రీనివాస్, మధుకర్, సైదులు, కొండా మధు, చంద్రశేఖర్ రెడ్డి, శ్రీనివాస్ రెడ్డి, యామిని, శ్రీనివాస్, నారాయణ స్వామి, రాజు, రామ్ మోహన్, శివదాస్, దూది, సుధాకర్, కుల్దీప్, నారయ, విజయ్ , శ్రీనివాస్, శీష్ రెడ్డి, ప్రవీణ్ కుమార్
--------------------------------------------------

Sample 2/3:
--------------------------------------------------
Generated: జగన్ మోహన్ రెడ్డిని చంద్రబాబు ప్రశ్నించారు బుధవారం విజయవాడలోని చంద్రబాబు నివాసంలో ఆయన విలేకరులతో మాట్లాడారు ప్రత్యేక హోదాపై రాజకీయ పోరాటం చేస్తే చంద్రబాబు చేతకానివాడని మండిపడ్డారు అన్ని పార్టీలు హోదా కోసం పోరాటం చేస్తాయన్నారు హోదా కోసం పోరాడుతున్న తెలుగుదేశం పార్టీ, ఇప్పుడు హోదా కోసం పోరాడుతోందని, ఈ పోరాటం వల్ల చంద్రబాబు నీతి, నిజాయితీ, నిజాయితీ లేని మనిషిగా మిగిలిపోయాడని ఎద్దేవా చేశారు బాబు కుటుంబ సభ్యులు తాను వచ్చి చంద్రబాబుని కలిసి నిరసన తెలిపి, ముందుకొచ్చి నిరసన తెలిపితే చంద్రబాబు స్పందించలేదన్నారు చంద్రబాబు చేతకానితనానికి చంద్రబాబు తీరు
--------------------------------------------------

Sample 3/3:
--------------------------------------------------
Generated: జగన్ మోహన్ రెడ్డి, కాకాని గోవర్దన్ రెడ్డి, కేవీఎం ప్రసాద్, రాష్ట్ర పర్యాటక శాఖ మంత్రి శ్రీనివాస రావు, ఏపీ పర్యాటక శాఖ మంత్రి నారా లోకేష్, మంత్రి బాలినేని శ్రీనివాస రెడ్డిలు చినబాబు, లోకేష్ లు పలువురు రాజకీయ నాయకులు, అధికారులు, మంత్రులు, ఎమ్మెల్యేలు, ఎంపీలు, నేతలు ఈ సమావేశంలో పాల్గొన్నారు ఈ నేపథ్యంలో ఇప్పుడు చినబాబు, లోకేష్ లు కూడా చినబాబును కలిశారు బీజేపీ, జనసేన నేతలు కూడా వచ్చే ఎన్నికల్లో చినబాబు బీజేపీని వదిలేస్తే బాగుంటుందని జగన్ భావిస్తున్నట్లు తెలుస్తోంది మరోవైపు
--------------------------------------------------
```
#### 10M, 1/10 of CulturaX 'te' data:
```
Generating 3 samples with 100 tokens each...
Optimized for A4000 GPU
================================================================================

Sample 1/3:
--------------------------------------------------
Generated: నమస్కారం, నేను తెలుగు-] ఇందులో చెందిన పెద్ద సినిమాకు 10] 141 జ్తో 3-3లో 24] 102-2 - చాలా ప్రపంచ ఎక్కువలులపై డెన్డిలు 1] 19-0 నుంచి టీడీపీ 125-టీ 15, 20 15న | 18 1/40, 1813- పోయి ఈ నెల, 163, ఇది
--------------------------------------------------

Sample 2/3:
--------------------------------------------------
Generated: నమస్కారం, నేను తెలుగుకి, ఇది ద్ "రెప్ యొక్క ఈ తర్వాత 51-1 శాతం మంది ఈ కు 20 వరకు 040, కడ్సి) 19 ఇక 30140196వణం దీనిపై జిల్లా 2, [2550010222462104400 [1040003013] 6 : ఐపిస్
--------------------------------------------------

Sample 3/3:
--------------------------------------------------
Generated: నమస్కారం, నేను తెలుగు రోజు in |] స్ అలు | 1)ఏ | march | 20 మిలియన్లు 54020/ రూ జిల్లా సంస్థలోని 150015 లక్షల పై 18559-17: 1520190-20195 pm నుండి oneindia 22013 pm 12027, 19:5, |90
--------------------------------------------------

================================================================================
A4000 Optimized Sampling Complete!
Final GPU memory: 0.42 GB
blockchain@amritesh-precision-3660:~/Insolare-LLM$ python3 sample.py
A4000 GPU Optimized Sampling
Using device: cuda
Using dtype: float16
Loading tokenizer from 10M.model...
Tokenizer loaded with vocab size: 16768
Loading checkpoint from out/trial_checkpoint.pth...
Checkpoint loaded to CPU
Model config: {'n_layer': 12, 'n_head': 12, 'n_embd': 768, 'block_size': 512, 'bias': False, 'dropout': 0.0, 'vocab_size': 16768}
Number of parameters:  98224896
Model created with 98,224,896 parameters
Model weights loaded successfully!
Moving model to cuda...
GPU memory allocated: 0.41 GB
GPU memory cached: 0.44 GB

Prompt: నమస్కారం, నేను తెలుగు
Encoded prompt: [13718, 265, 371, 416]
Prompt length: 4 tokens

Generating 3 samples with 100 tokens each...
Optimized for A4000 GPU
================================================================================

Sample 1/3:
--------------------------------------------------
Generated: నమస్కారం, నేను తెలుగు ఆర్థిక రోజులు విధాల్సివిస్తోంది ఆ తో ఉంది అయినప్పటికీ అన్న ’ అనే జంటలోని ‘స్తుస్ 15 మందికి నాకు మూడు |19 ఉన్నాయి 16 నుంచి 120] telugu [0-213) –13 లో ఆ చిత్రం 144 నుంచి కూడా 2, 3205 నుంచి 20190008348
--------------------------------------------------

Sample 2/3:
--------------------------------------------------
Generated: నమస్కారం, నేను తెలుగు జీవితాన్ని నా 13 వసి 2, కు 1, 1: ఎస్ 3 - 08 కరోనా లో కలిసి ఆ మధ్య తనది jle 13015-20 నుండి 10] 149నోగా మాత్రమే | 30-3, 113 లో ఓ రెండు ప్రైవేటు తల్లికు తిరిగి - లు, ఉప్పు, 198-2206:15
--------------------------------------------------

Sample 3/3:
--------------------------------------------------
Generated: నమస్కారం, నేను తెలుగు దేశంాజీకు ఉంది, కూడా మాట్లాడుకుని ఉన్న సమయంలో ఈ రాష్ట్ర రకాల ఎక్కువగా అని అడుగుకు కూడా ఉన్న భావంలను ఈ రోజు, ఆ "కే 20018-48 నేను ఏ లో, ఎక్కువ అనే కు కేంద్ర 100 లక్షల 3 hid news 19215, 1020 1, 3 3 మంది telugu: 158993] 14వేల అని ఉన్న రోజుల -
--------------------------------------------------

```

### Useful commands
* `python3 download.py`
* `nohup python3 prepare_data.py > /dev/null 2>&1 &`
* `nohup python3 train_tokenizer.py --input_file telugu_training_data_full.txt --model_prefix 16768_full_txt --vocab_size 16768 > tokenizer_16768.log 2>&1 &`
* `python3 txt_to_bin.py`
* `nohup python3 train.py > train.log 2>&1`

### BTS (behind the scenes)
* DOT stands for Decoder-only-Transformer, a funny pun intended for a dialogue in Shanker's Robo (2010) movie.
* I recently trained the next version of Buggi (an SLM on Telugu text): 10M, 98M, and 124M parameter model variants (a modified GPT-2 architecture) on CulturaX's Telugu collection. Learnt a lot about processing raw text, training a tokeniser, training dynamics, loss functions, hyperparameters, and GPUs at a relatively bigger scale than Buggi-1. Ran a few experiments on a smaller variant and a subset of data, then gradually scaled it up, and experienced the capability grow from generating short, valid words to sentences to paragraphs. Turns out, with good textual data, transformer-based architecture, tokeniser, compute, one could just train a model to memorise/encode any data (kinda by heart) as its parameters. Since a lot of data is news articles about movies, mostly, a noun is contextualised as a movie name by the model 🤣, for ex, Amaravati (Andhra Pradesh capital) is confused as a movie name when I asked about its progress. Will host it on GCP if I get to use GCP's 300$ credit with GPU for sometime. See sample results: github.com/Vasudeva-bit/buggi - posted on @KilaruVasudeva

### More data for Telugu
* Swetcha (HYD NGO), AI4Bharat (IIT-Madras)
* A few other: LLM Labs, Anuskha/Anusha Kaggle Telugu Books etc...
