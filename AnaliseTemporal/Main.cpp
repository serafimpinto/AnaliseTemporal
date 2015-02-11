#include <iostream>
#include "Functions.h"

int main(int argc, char** argv) {
	while (true){
		int x = -1;
		printf("\n1 para testar camera\n0 para sair\n");
		scanf_s("%d", &x);

		switch (x)
		{
		case 1:
			captureVideo();
			break;
		case 0:
			return 0;
			break;
		default:
			break;
		}

	}
	return 0;
}