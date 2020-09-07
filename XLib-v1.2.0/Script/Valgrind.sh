
#apply Valgrind

valgrind --show-leak-kinds=all --leak-check=full --track-origins=yes --suppressions=./cuda.supp --max-stackframe=2818064 "$@" "$@"


valgrind --show-leak-kinds=all --leak-check=full --track-origins=yes\
         --gen-suppressions=all --log-file=valgrind.log --max-stackframe=2818064 "$@"

--track-fds=yes
