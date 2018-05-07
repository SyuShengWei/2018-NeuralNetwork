#include <iostream>
#include <math.h>
#include <time.h>
#include <stdlib.h>

using namespace std;


#define IL 3 //inputLayer
#define HL 3 //hiddenLayer
#define OL 1 //outpurLayer

double sigmoid(double x){
	return 1.0/(1.0 + exp(-x));
}

double sigmoid_prime(double x){
	return sigmoid(x)*(1.0-sigmoid(x));
}

class BPN{

private:
	int inputLayer;
	int hiddenLayer;
	int outputLayer;

	double eta;
	double alpha;

	double 	Xk[IL];
	double	Zh[HL];
	double	SZh[HL];
	double	Yj[OL];
	double	SYj[OL];

	double delta_J[OL];
	double delta_Whj[HL][OL];
	double delta_H[HL];
	double delta_Wih[IL][HL-1];

	double delta_Wih_last[IL][HL-1];
	double delta_Whj_last[HL][OL];


public:
	BPN(double the_eta, double the_alpha);

	double Wih[IL][HL-1]; 	// 3x2
	double Whj[HL][OL];		// 3x1

	void forward(double X[2]);
	void backward(double X[2],double D);

	double getSYj(){ return this->SYj[0];};
};


BPN::BPN(double the_eta, double the_alpha){
	this->inputLayer  = IL;
	this->hiddenLayer = HL;
	this->outputLayer = OL;

	this->eta = the_eta;
	this->alpha = the_alpha;

	srand(time(NULL));
	for(int i = 0 ; i < IL ; i++){
		for(int h = 0 ; h < HL-1 ; h++){
			this->Wih[i][h] = (double) rand() / (RAND_MAX + 1.0 );
		}
	}

	for (int h = 0 ; h < HL ; h++){
		for(int j = 0 ; j < OL ; j++){
			this->Whj[h][j] = (double) rand() / (RAND_MAX + 1.0 );
		}
	}

	for(int i = 0 ; i < IL ; i++){
		for(int h = 0 ; h < HL-1 ; h++){
			this->delta_Wih_last[i][h] = 0;
		}
	}

	for (int h = 0 ; h < HL ; h++){
		for(int j = 0 ; j < OL ; j++){
			this->delta_Whj_last[h][j] = 0;
		}
	}

}

void BPN::forward(double X[2]){
	//initial Xk with X0
	this->Xk[0] = 1.0;//input bias
	for(int i = 1 ; i < IL ; i++){ this->Xk[i] = X[i-1];}
	//for(int i = 0 ; i < IL ; i++){ cout<<this->Xk[i]<<" ";}
	//cout<<endl;
	
	//count Zh
	this->Zh[0] = 0;
	for(int h = 1 ; h < HL ; h++){
		double sum_up = 0;
		for(int i = 0 ; i < IL ; i++){
			sum_up += this->Wih[i][h-1]*this->Xk[i];
		}
		this->Zh[h] = sum_up;
	}
	//cout<<endl;
	//for(int h = 1 ; h < HL ; h++){
	//	for(int i = 0 ; i < IL ; i++){
	//		cout<<Wih[i][h-1]<<" ";
	//	}
	//	cout<<endl;
	//}
	//cout<<endl;
	//for(int h = 0 ; h < HL ; h++){ cout<<this->Zh[h]<<endl;}
	
	//count SZh 
	this->SZh[0] = 1;
	for(int h = 1 ; h < HL ; h++){ this->SZh[h] = sigmoid(this->Zh[h]); }
	//for(int h = 0 ; h < HL ; h++){ cout<<this->SZh[h]<<endl;}

	//count Yj
	for(int j = 0 ; j < OL ; j++){
		double sum_up = 0;
		for (int h = 0 ; h < HL ; h++){
			sum_up += this->Whj[h][j]*this->SZh[h];
		}
		this->Yj[j] = sum_up;
	}

	//count SYj
	for(int j = 0 ; j < OL ; j++){ this->SYj[j] = sigmoid(this->Yj[j]); }
	//for(int j = 0 ; j < OL ; j++){ cout<<this->SYj[j]<<endl; }
}

void BPN::backward(double X[2],double D){
	//count delta_J
	for(int j = 0 ; j < OL ; j++){
		this->delta_J[j] = (D - this->SYj[j])*sigmoid_prime(this->Yj[j]);
	}

	//count delta_Whj
	for (int h = 0 ; h < HL ; h++){
		for(int j = 0 ; j < OL ; j++){
			this->delta_Whj[h][j] = this->eta * this->delta_J[j] * this->Xk[h];
		}
	}

	//count delta_H
	for(int h = 0 ; h < HL ; h++){
		double sum_up = 0;
		for(int j = 0 ; j < OL ; j++){
			sum_up += delta_J[j]*this->Whj[h][j];
		} 
		this->delta_H[h] = sum_up * sigmoid_prime(this->Zh[h]);
	}

	//count delta_Wjh
	for(int i = 0 ; i < IL ; i++){
		for(int h = 1 ; h < HL ; h++){
			this->delta_Wih[i][h-1] = this->eta * this->delta_H[h]* this->Xk[i];
		}
	}

	//Update weight
	for(int i = 0 ; i < IL ; i++){
		for(int h = 0 ; h < HL-1 ; h++){
			this->Wih[i][h] += this->delta_Wih[i][h] ;
		}
	}

	for (int h = 0 ; h < HL ; h++){
		for(int j = 0 ; j < OL ; j++){
			this->Whj[h][j] += delta_Whj[h][j];
		}
	}

	//Record delta_W
	for(int i = 0 ; i < IL ; i++){
		for(int h = 0 ; h < HL-1 ; h++){
			this->delta_Wih_last[i][h] = delta_Wih[i][h];
		}
	}

	for (int h = 0 ; h < HL ; h++){
		for(int j = 0 ; j < OL ; j++){
			this->delta_Whj_last[h][j] = delta_Whj[h][j];
		}
	}

}


int main (){
	double X[4][2] = {{0.05,0.05},
				  {0.05,0.95},
				  {0.95,0.05},
				  {0.95,0.95}};
	double D[4]	   = {0.05,0.95,0.95,0.05};

	for (int i = 0 ; i < 4 ; i ++){
		cout<<X[i][0]<<" "<<X[i][1]<<" : "<<D[i]<<endl;
	}

	BPN* bpn = new BPN(0.2,0.2);
	
	int epcho_ctr = 0;

	double error_avg = 100000;
	while (error_avg > 0.001){
		double error_sum = 0;
		for(int k = 0 ; k < 4 ; k++){
			bpn->forward(X[k]);
			bpn->backward(X[k],D[k]);
			double the_SYj = bpn->getSYj();
			error_sum += (the_SYj - D[k])*(the_SYj - D[k]);
		}
		epcho_ctr+=1;
		error_avg = error_sum /(2*4);
		cout<<"epcho : "<< epcho_ctr<<" error_ave : " <<error_avg<<endl;
	}

	for(int k = 0 ; k < 4 ; k++){
		bpn->forward(X[k]);
		cout<<"X : "<<X[k][0]<<" , "<< X[k][1]<<", D : "<<D[k]<<", Y(BPN) : "<<bpn->getSYj()<<endl;
	}




	//cout<<sigmoid(1);
	return 0;
}