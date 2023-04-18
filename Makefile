# ---------------------------------------------------------
# Yuxuan Zhang
# Dept. of Electrical and Computer Engineering
# University of Florida
# ---------------------------------------------------------
# Scripts for convience of use
# ---------------------------------------------------------
%.zip:
	zip -r $@ run

zip: var/run.zip

.PHONY: %.zip zip run/%