CC=mcs

TARGET=XGBoostSharp

all: $(TARGET).exe

$(TARGET).exe: $(TARGET).cs
	cp ../cpp/libxgboost_sharp.so ./
	$(CC)  $(TARGET).cs -out:$(TARGET).exe

clean:
	$(RM) $(TARGET).exe
