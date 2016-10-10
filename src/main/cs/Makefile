CC=mcs

LIBDIR=../../../lib
BINDIR=../../../bin

TARGET=XGBoostSharp

all: $(TARGET).exe

$(TARGET).exe: $(TARGET).cs
	cp $(LIBDIR)/libxgboost_sharp.so ./
	$(CC)  $(TARGET).cs -out:$(BINDIR)/$(TARGET).exe

clean:
	$(RM) $(BINDIR)/$(TARGET).exe
