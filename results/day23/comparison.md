# Day 23 — PyTorch vs TensorFlow evaluation

Shared validation examples: 100

## Metrics

| Framework | BLEU-4 | BP | Length ratio | Decode seconds | Sentences/s |
|---|---:|---:|---:|---:|---:|
| PyTorch | 37.8505 | 0.9795 | 0.9798 | 51.10 | 1.957 |
| TensorFlow | 1.6422 | 1.0000 | 1.2710 | 364.34 | 0.274 |

BLEU is lowercase corpus BLEU-4 using the repository's Unicode word/punctuation tokenizer and exponential zero-count smoothing.

## Qualitative examples

| Source | Reference | PyTorch | TensorFlow |
|---|---|---|---|
| A group of men are loading cotton onto a truck | Eine Gruppe von Männern lädt Baumwolle auf einen Lastwagen | eine gruppe von männern lässt sich auf einen lkw an . | ein junge mit einem roten hemd , der auf einem bürgersteig spielt . |
| A man sleeping in a green room on a couch. | Ein Mann schläft in einem grünen Raum auf einem Sofa. | ein mann schläft in einem grünen zimmer auf einem sofa . | ein mann mit einem roten hemd , der einen kopf auf dem boden . |
| A boy wearing headphones sits on a woman's shoulders. | Ein Junge mit Kopfhörern sitzt auf den Schultern einer Frau. | ein junge mit kopfhörern sitzt auf den schultern einer frau . | ein junge mit einem roten hemd und einem roten hemd , der auf einem bürgersteig steht . |
| Two men setting up a blue ice fishing hut on an iced over lake | Zwei Männer bauen eine blaue Eisfischerhütte auf einem zugefrorenen See auf | zwei männer bauen eine blaue eiswaffeln auf einem abgesperrten see . | zwei männer , die auf einem bürgersteig in einem keig in einem gebäude , der nähe eines mannes . |
| A balding man wearing a red life jacket is sitting in a small boat. | Ein Mann mit beginnender Glatze, der eine rote Rettungsweste trägt, sitzt in einem kleinen Boot. | ein mann mit einer roten schwimmweste sitzt in einem kleinen boot . | ein mann mit einem roten hemd und einem roten hemd , der einen kopf auf dem boden . |
| A lady in a red coat, holding a bluish hand bag likely of asian descent, jumping off the ground for a snapshot. | Eine Frau in einem rotem Mantel, die eine vermutlich aus Asien stammende Handtasche in einem blauen Farbton hält, springt für einen Schnappschuss in die Luft. | eine dame in einem roten mantel hält eine blaue tasche in der hand oder hochzeitskleidung eines asiatischen studio und springt von einem schiedsrichter . | ein mann mit einem roten hemd und einem schwarzen hemd , der auf einem bürgersteig in einem gebäude sitzt . |
| A brown dog is running after the black dog. | Ein brauner Hund rennt dem schwarzen Hund hinterher. | ein brauner hund rennt hinter dem schwarzen hund . | ein brauner hund , der im gras rennt . |
| A young boy wearing a Giants jersey swings a baseball bat at an incoming pitch. | Ein kleiner Junge mit einem Giants-Trikot schwingt einen Baseballschläger in Richtung eines ankommenden Balls. | ein kleiner junge in einem riesigen trikot schwingt einen baseballschläger an einem kommenden wurf . | ein junge mit einem roten hemd und einem roten hemd , der auf einem bürgersteig in einem schlafen . |
| A man in a cluttered office is using the telephone | Ein Mann telefoniert in einem unaufgeräumten Büro | ein mann in einem kreuzten büro benutzt das telefon . | ein mann mit einem roten hemd und einem roten hemd , der einen kopf auf dem boden . |
| A smiling woman in a peach tank top stands holding a mountain bike | Eine lächelnde Frau mit einem pfirsichfarbenen Trägershirt hält ein Mountainbike | eine lächelnde frau in einem papieroberteil steht mit einem mountainbike und hält einen mountainbike . | ein mann mit einem roten hemd und einem roten hemd , der einen kopf auf einem bürgersteig fen . |
