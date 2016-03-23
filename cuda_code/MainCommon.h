#ifndef MAIN_COMMON
#define MAIN_COMMON

#include<cstdlib>
#include<cstdio>

// Forward declarations.
class Mod4;
template<int> class Mod;
template<class> void test(const unsigned int);

int main(int, char**);

int main(int argc, char** argv) {
    if (argc < 2) {
        printf("The program takes <num_elems> as argument!\n");
        return EXIT_FAILURE;
    }
    
    const unsigned int num_elems = strtoul(argv[1], NULL, 10);
    
#ifdef THRUST
    printf("Using Thrust scan.\n");
#else
    printf("Using custom scan.\n");
#endif
    
    if (argc == 2) {
        printf("Mod4:\n");
        test<Mod4>(num_elems);
    }
    else {
        int num = strtol(argv[2], NULL, 10);
        switch (num) {
        case 1 :
            printf("Mod<1>:\n");
            test< Mod<1> >(num_elems);
            break;
        case 2 :
            printf("Mod<2>:\n");
            test< Mod<2> >(num_elems);
            break;
        case 3 :
            printf("Mod<3>:\n");
            test< Mod<3> >(num_elems);
            break;
        case 4 :
            printf("Mod<4>:\n");
            test< Mod<4> >(num_elems);
            break;
        case 5 :
            printf("Mod<5>:\n");
            test< Mod<5> >(num_elems);
            break;
        case 6 :
            printf("Mod<6>:\n");
            test< Mod<6> >(num_elems);
            break;
        case 7 :
            printf("Mod<7>:\n");
            test< Mod<7> >(num_elems);
            break;
        case 8 :
            printf("Mod<8>:\n");
            test< Mod<8> >(num_elems);
            break;
        default :
            printf("Unsupported number of equivalence classes.\n");
            return EXIT_FAILURE;
        }
    }
    
    return EXIT_SUCCESS;
}

#endif //MAIN_COMMON
