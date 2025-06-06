### Brain Dump:
From what I understand we are transferring a CLIP model for obtaining embeddings for an image with a caption, and with those image + text embeddings, we are passing that through a custom built decoder to predict the sequence of text that makes up a caption.
-> Build a decoder with masked self-attention (alt. Cross-attn (this is worse + harder though))

-> Train this decoder with "what loss function"? (softmax activation should tell me that I'm doing a classification problem and therefore want to use cross-entropy loss?)

-> how does inference work? I know that there is a sequence of patches passed into CLIP, then we get some embeddings with dimension of 768 (is recommended, this can be experimented with but low priority).
	-> The patches are passed into the decoder along with `<Sep><Start>` tokens appended
	-> The decoder then uses attention to build up a weighted contextual embedding that is used to predict the next word ?
	-> That predicted word is then appended to the sequence (patches + tokens + generated word) and the next word is predicted (iterate as many times as needed, end with `<Stop>` token).
The core difference between this task and the last one is that the path we are following structures the decoder to take in the flattened image patches alongside the `<START>` token and uses self-attention  across all of that info to produce a word prediction (repeat till `<STOP>` token produced.)
