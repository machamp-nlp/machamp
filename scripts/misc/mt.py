from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

#en_text = "Do not meddle in the affairs of wizards, for they are subtle and quick to anger."
fi_text = "Älä sekaannu velhojen asioihin, sillä ne ovat hienovaraisia ja nopeasti vihaisia."

tokenizer = AutoTokenizer.from_pretrained("facebook/mbart-large-50-many-to-many-mmt", src_lang="fi_FI")
model = AutoModelForSeq2SeqLM.from_pretrained("facebook/mbart-large-50-many-to-many-mmt")

encoded_en = tokenizer(fi_text, return_tensors="pt")

generated_tokens = model.generate(**encoded_en, forced_bos_token_id=tokenizer.lang_code_to_id["en_XX"])
print(tokenizer.batch_decode(generated_tokens, skip_special_tokens=True))

