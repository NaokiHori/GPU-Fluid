CC     := clang
CFLAG  := -std=c99 -Wall -Wextra -Wpedantic -O3
INC    := -Iinclude
LIB    := -lm -framework Metal -framework Foundation -framework MetalPerformanceShaders -framework MetalPerformanceShadersGraph
SRCDIR := src
OBJDIR := obj
SRCS   := $(shell find $(SRCDIR) -type f \( -name "*.c" -o -name "*.m" \))
OBJS   := $(patsubst $(SRCDIR)/%, $(OBJDIR)/%.o, $(SRCS))
DEPS   := $(patsubst %.o, %.d, $(OBJS))
TARGET := a.out
OUTDIR := output

help:
	@echo "all     : create \"$(TARGET)\""
	@echo "clean   : remove \"$(TARGET)\" and object files under \"$(OBJDIR)\""
	@echo "output  : create \"$(OUTDIR)\""
	@echo "datadel : clean-up \"$(OUTDIR)\""
	@echo "help    : show this message"

all: $(TARGET)

clean:
	$(RM) -r $(OBJDIR) $(TARGET)

$(TARGET): $(OBJS)
	$(CC) $(CFLAG) -o $@ $^ $(LIB)

$(OBJDIR)/%.c.o: $(SRCDIR)/%.c
	@mkdir -p $(dir $@)
	$(CC) $(CFLAG) -MMD $(INC) -c $< -o $@

$(OBJDIR)/%.m.o: $(SRCDIR)/%.m
	@mkdir -p $(dir $@)
	$(CC) $(CFLAG) -MMD $(INC) -c $< -o $@

output:
	@mkdir -p $(OUTDIR)/image;
	@mkdir -p $(OUTDIR)/log;
	@mkdir -p $(OUTDIR)/save;

datadel:
	@rm -rf $(OUTDIR)
	@make output

-include $(DEPS)

.PHONY : help all clean datadel output
