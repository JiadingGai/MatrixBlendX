TEX := fa2_deep_dive
PDFLATEX := pdflatex
PDFLATEX_FLAGS := -interaction=nonstopmode -halt-on-error

.PHONY: all pdf open clean

all: pdf

pdf: $(TEX).pdf

open: pdf
	open $(TEX).pdf

$(TEX).pdf: $(TEX).tex
	$(PDFLATEX) $(PDFLATEX_FLAGS) $(TEX).tex
	$(PDFLATEX) $(PDFLATEX_FLAGS) $(TEX).tex

clean:
	rm -f $(TEX).aux $(TEX).log $(TEX).out $(TEX).pdf
