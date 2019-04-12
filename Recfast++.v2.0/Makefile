#===============================================================================
# set to C++ compiler that should be used
#===============================================================================
CC = g++

#===============================================================================
# compiler and linker flags
#===============================================================================
CXXFLAGS = -Wall -O2
CXXFLAGSLOC = $(CXXFLAGS) -D RECFASTPPPATH=\"$(PWD)/\"
LXXFLAGS =

#===============================================================================
# object-files
#===============================================================================
OBJSLIB = ./Recfast++.o \
	   ./src/cosmology.Recfast.o \
	   ./src/evalode.Recfast.o \
	   ./src/ODE_solver.Recfast.o \
	   ./src/DM_annihilation_decay.Recfast.o \
	   ./src/Magnetic_fields.Recfast.o \
	   ./src/Variation_constants.Recfast.o \
	   ./src/Rec_corrs_CT.Recfast.o

OBJS = ./src/parser.Recfast.o

#===============================================================================
# program	 
#===============================================================================
all: Recfast++lib Recfast++

Recfast++:  $(OBJS) libRecfast++.a main_Recfast++.o
			 @echo "Linking..."
			 $(CC) $(LXXFLAGS) main_Recfast++.o -L. -lRecfast++ $(OBJS) -o Recfast++

lib: Recfast++lib

Recfast++lib: $(OBJSLIB)
			 @echo "\n Creating Recfast++ lib library\n"
			 ar rvs ./libRecfast++.a $?

pub:
	make -f Makefile.pub all

pubtar:
	make -f Makefile.pub tarball

clean:
	rm -f ./src/*.o

tidy:
	rm -f ./src/*.o ./src/*.o~ 
	rm -f *.o *~ Recfast++ libRecfast++.a

wipeDS:
	find . -type f -name \.DS_Store -print | xargs rm

#===============================================================================
# rules
#===============================================================================
INC_PATH = -I./include -I.
		   
.cpp.o:
	@echo "Producing object-file $@"
	$(CC) $(CXXFLAGSLOC) $(INC_PATH) -c $< -o $@
	
#===============================================================================
#===============================================================================
