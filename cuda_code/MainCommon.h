#ifndef MAIN_COMMON
#define MAIN_COMMON

#include<cstdlib>
#include<cstdio>

// Forward declarations.
class Mod4;
template<unsigned int> class Mod;
template<class> int test(const unsigned int);

/* int main(int argc, char** argv) { */
/*     if (argc < 2) { */
/*         printf("The program takes <num_elems> as argument!\n"); */
/*         return EXIT_FAILURE; */
/*     } */
    
/*     const unsigned int num_elems = strtoul(argv[1], NULL, 10); */
    
/* #ifdef THRUST */
/*     fprintf(stderr, "Using Thrust scan\n"); */
/* #else */
/*     fprintf(stderr, "Using custom scan\n"); */
/* #endif */
    
/*     if (argc == 2) { */
/*         fprintf(stderr, "Discriminator is Mod4\n"); */
/*         return test<Mod4>(num_elems); */
/*     } */
/*     else { */
/*         int num = strtol(argv[2], NULL, 10); */
/*         switch (num) { */
/*         case 1 : */
/*             fprintf(stderr, "Discriminator is Mod<1>\n"); */
/*             return test< Mod<1> >(num_elems); */
/*         case 2 : */
/*             fprintf(stderr, "Discriminator is Mod<2>\n"); */
/*             return test< Mod<2> >(num_elems); */
/*         case 3 : */
/*             fprintf(stderr, "Discriminator is Mod<3>\n"); */
/*             return test< Mod<3> >(num_elems); */
/*         case 4 : */
/*             fprintf(stderr, "Discriminator is Mod<4>\n"); */
/*             return test< Mod<4> >(num_elems); */
/*         case 5 : */
/*             fprintf(stderr, "Discriminator is Mod<5>\n"); */
/*             return test< Mod<5> >(num_elems); */
/*         case 6 : */
/*             fprintf(stderr, "Discriminator is Mod<6>\n"); */
/*             return test< Mod<6> >(num_elems); */
/*         case 7 : */
/*             fprintf(stderr, "Discriminator is Mod<7>\n"); */
/*             return test< Mod<7> >(num_elems); */
/*         case 8 : */
/*             fprintf(stderr, "Discriminator is Mod<8>\n"); */
/*             return test< Mod<8> >(num_elems); */
/*         case 9 : */
/*             fprintf(stderr, "Discriminator is Mod<9>\n"); */
/*             return test< Mod<9> >(num_elems); */
/*         default : */
/*             printf("Unsupported number of equivalence classes.\n"); */
/*             return EXIT_FAILURE; */
/*         } */
/*     } */
/* } */

int main(int argc, char** argv) {
    if (argc != 2) {
        printf("The program takes <num_elems> as argument!\n");
        return EXIT_FAILURE;
    }
    
    const unsigned int num_elems = strtoul(argv[1], NULL, 10);
    
#ifdef THRUST
    fprintf(stderr, "Using Thrust scan\n");
#else
    fprintf(stderr, "Using custom scan\n");
#endif

#ifndef NUM_CLASSES
#error NUM_CLASSES not defined!
#else
#ifdef SPECIALIZED
#if NUM_CLASSES == 4
    fprintf(stderr, "Discriminator is Mod4\n");
    return test<Mod4>(num_elems);
#elif NUM_CLASSES == 8
    fprintf(stderr, "Discriminator is Mod8\n");
    return test<Mod8>(num_elems);
#else
#error Unsupported number of equivalence classes!
#endif
#else
    fprintf(stderr, "Discriminator is Mod<%d>\n", NUM_CLASSES);
    return test< Mod<NUM_CLASSES> >(num_elems);
#endif
#endif
}

#endif //MAIN_COMMON
