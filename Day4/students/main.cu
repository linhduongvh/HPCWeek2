/*
* LW 5 - Atomics
* --------------------------
* Histogram equalization
*
* File: main.cu
*/

#include "app.hpp"

int main( int argc, char **argv ) {
	App *app = App::createInstance();
	app->launch( argc, argv, false );
	return ( EXIT_SUCCESS );
}
