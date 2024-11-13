# oak-spacy-demo

Comparing annotations between oaklib and spacy.

## OAK

example CLI command to run oak:
```bash
osd annotate --tool oak --input-file tests/input/test_diseases.tsv --column label -o abcd.tsv --resource data/mondo.owl
```


## SciSpacy

[Models](https://allenai.github.io/scispacy/)
 - en_core_sci_sm: https://s3-us-west-2.amazonaws.com/ai2-s2-scispacy/releases/v0.5.4/en_core_sci_sm-0.5.4.tar.gz
 - en_core_sci_md: https://s3-us-west-2.amazonaws.com/ai2-s2-scispacy/releases/v0.5.4/en_core_sci_md-0.5.4.tar.gz
 - en_core_sci_scibert: https://s3-us-west-2.amazonaws.com/ai2-s2-scispacy/releases/v0.5.4/en_core_sci_scibert-0.5.4.tar.gz
 - en_core_sci_lg: https://s3-us-west-2.amazonaws.com/ai2-s2-scispacy/releases/v0.5.4/en_core_sci_lg-0.5.4.tar.gz
 - en_ner_craft_md: https://s3-us-west-2.amazonaws.com/ai2-s2-scispacy/releases/v0.5.4/en_ner_craft_md-0.5.4.tar.gz
 - en_ner_jnlpba_md: https://s3-us-west-2.amazonaws.com/ai2-s2-scispacy/releases/v0.5.4/en_ner_jnlpba_md-0.5.4.tar.gz
 - en_ner_bc5cdr_md: https://s3-us-west-2.amazonaws.com/ai2-s2-scispacy/releases/v0.5.4/en_ner_bc5cdr_md-0.5.4.tar.gz
 - en_ner_bionlp13cg_md: https://s3-us-west-2.amazonaws.com/ai2-s2-scispacy/releases/v0.5.4/en_ner_bionlp13cg_md-0.5.4.tar.gz