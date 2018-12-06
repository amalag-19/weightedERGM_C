// -*- mode: C++; c-indent-level: 4; c-basic-offset: 4; indent-tabs-mode: nil; -*-

// we only include RcppArmadillo.h which pulls Rcpp.h in for us
#include "RcppArmadillo.h"
using namespace arma;
using namespace std;

// via the depends attribute we tell Rcpp to create hooks for
// RcppArmadillo so that the build process will know what to do
//
// [[Rcpp::depends(RcppArmadillo)]]

// simple example of creating two matrices and
// returning the result of an operatioon on them
//
// via the exports attribute we tell Rcpp to make this function
// available from R

///////////////////////////////////////////////////////////////////////////////////////////////////////////////

// [[Rcpp::export]]
vec rowsum_Mat(mat M) {
    int nr=M.n_rows;
    vec out(nr);
    for(int i=0;i<nr;i++){
        out(i)=sum(M.row(i));
    }
    return out;
}

// [[Rcpp::export]]
vec colsum_Mat(mat M) {
    int nc=M.n_cols;
    vec out(nc);
    for(int i=0;i<nc;i++){
        out(i)=sum(M.col(i));
    }
    return out;
}

// [[Rcpp::export]]
float epan(float input){
    float output;
    if(abs(input)<=1){
        output=0.75*(1-pow(input,2));
    }
    else{
        output=0;
    }
    return output;
}

///////////////////////////////////////////////////////////////////////////////////////////////////////////////
// gamma, Tau update function, gradient and hessian functions and ELBO convergence function

// [[Rcpp::export]]
cube gamma_update_weighted_stat_undir(mat gamma, vec pi, vec theta, mat block_dens_mat, mat net_adjacency, int N, int K){
    mat exp_val_ratio_mat(K,K);
    for(int k = 0; k < K; k++){
        for(int l = 0; l < K; l++){
            float exp_val=exp(theta(k)+theta(l));
            exp_val_ratio_mat(k,l)=exp_val/(1+exp_val);
        }
    }
    cube quad_lin_coeff(N,K,2);
    for(int i = 0; i < N; i++){
        if(i!=(N-1)){
            for(int k = 0; k < K; k++){
                float t1=0;
                for(int j = i+1; j < N; j++){
                    for(int l = 0; l < K; l++){
                        int indicator_0=(net_adjacency(i,j)==0);
                        if(indicator_0){
                            t1+=(gamma(j,l)/(2*gamma(i,k)))*(log(1-exp_val_ratio_mat(k,l)));
                        }else{
                            t1+=(gamma(j,l)/(2*gamma(i,k)))*(log(exp_val_ratio_mat(k,l))+log(block_dens_mat((min(k,l)*N+i),(max(k,l)*N+j))));
                        }
                    }
                }
                quad_lin_coeff(i,k,0)=t1-(1/gamma(i,k));
                quad_lin_coeff(i,k,1)=log(pi(k))-log(gamma(i,k))+1;
            }
        } else if(i==(N-1)){
            for(int k = 0; k < K; k++){
                quad_lin_coeff(i,k,0)=-(1/gamma((N-1),k));
                quad_lin_coeff(i,k,1)=log(pi(k))-log(gamma((N-1),k))+1;
            }
        }
    }
    return quad_lin_coeff;
}

// [[Rcpp::export]]
mat grad_theta_weighted_stat_undir(vec theta, mat gamma, mat net_adjacency, int N, int K){
    vec grad_vector(K);
    mat grad_mat(K,K,fill::zeros);
    for(int i = 0; i < (N-1); i++){
        for(int j = i+1; j < N; j++){
            mat grad_matsub(K,K);
            for(int k = 0; k < K; k++){
                for(int l = 0; l < K; l++){
                    float exp_val=exp(theta(k)+theta(l));
                    bool indicator_0=(net_adjacency(i,j)==0);
                    grad_matsub(k,l)=gamma(i,k)*gamma(j,l)*(!indicator_0-(exp_val/(1+exp_val)));
                }
            }
            grad_mat+=grad_matsub;
        }
    }
    vec rsum=rowsum_Mat(grad_mat);
    vec csum=colsum_Mat(grad_mat);
    for(int k = 0; k < K; k++){
        grad_vector(k)=rsum(k)+csum(k);
    }
    return grad_vector;
}

// [[Rcpp::export]]
mat hess_theta_weighted_stat_undir(vec theta, mat gamma, int N, int K){
    mat t1(K,K);
    mat hess_mat(K,K,fill::zeros);
    for(int i = 0; i < (N-1); i++){
        for(int j = i+1; j < N; j++){
            mat hess_matsub(K,K);
            for(int k = 0; k < K; k++){
                for(int l = 0; l < K; l++){
                    float exp_val=exp(theta(k)+theta(l));
                    hess_matsub(k,l)=-(gamma(i,k)*gamma(j,l)*(exp_val/pow((1+exp_val),2)));
                }
            }
            hess_mat+=hess_matsub;
        }
    }
    for(int k = 0; k < K; k++){
        for(int l = 0; l < K; l++){
            if(k!=l){
                t1(k,l)=hess_mat(k,l)+hess_mat(l,k);
            }
        }
    }
    vec rsum=rowsum_Mat(hess_mat);
    vec csum=colsum_Mat(hess_mat);
    for(int k = 0; k < K; k++){
        t1(k,k)=(csum(k)+rsum(k)+(2*hess_mat(k,k)));
    }
    return t1;
}

// Getting the y_ij for all piars of clusters
// [[Rcpp::export]]
field<mat> tie_clust_partition(vec clust_est, mat net_adjacency, mat net_weight, int N, int K){
    field<mat> F(K,K);
    for(int k = 0; k < K; k++){
        for(int l = k; l < K; l++){
            vec ties_kl((N*(N-1))/2);
            ties_kl.fill(NA_REAL);
            int index_kl=0;
            for (int i=0; i<(N-1); i++){
                if(clust_est(i)==(k+1)){
                    for(int j=i+1; j<N; j++){
                        if(clust_est(j)==(l+1)){
                            if(net_adjacency(i,j)!=0){
                                ties_kl(index_kl)=net_weight(i,j);
                                index_kl+=1;
                            }
                        }
                    }
                }
            }
            F(k,l) = ties_kl;
        }
    }
    return F;
}


//////// [[Rcpp::export]]
//field<mat> checker_1(vec u, vec v, vec w, vec x){
//    field<mat> F(2,2);
//    F(0,0)=u;
//    F(1,0)=v;
//    F(0,1)=w;
//    F(1,1)=x;
//    return F;
//}

// [[Rcpp::export]]
float ELBO_conv_weighted_stat_undir(mat gamma, vec pi, vec theta, mat block_dens_mat, mat net_adjacency, int N, int K){
    mat exp_val_ratio_mat(K,K);
    for(int k = 0; k < K; k++){
        for(int l = 0; l < K; l++){
            float exp_val=exp(theta(k)+theta(l));
            exp_val_ratio_mat(k,l)=exp_val/(1+exp_val);
        }
    }
    float t1=0;
    for(int i = 0; i < (N-1); i++){
        for(int j = i+1; j < N; j++){
            for(int k = 0; k < K; k++){
                for(int l = 0; l < K; l++){
                    int indicator_0=(net_adjacency(i,j)==0);
                    if(indicator_0){
                        t1+=(gamma(i,k)*gamma(j,l)*(log(1-exp_val_ratio_mat(k,l))));
                    }else{
                        t1+=(gamma(i,k)*gamma(j,l)*(log(exp_val_ratio_mat(k,l))+log(block_dens_mat((min(k,l)*N+i),(max(k,l)*N+j)))));
                    }
                }
            }
        }
    }
    float t2=0;
    for(int i = 0; i < N; i++){
        for(int k = 0; k < K; k++){
            if((pi(k)>=(pow(10,(-100))))&(gamma(i,k)>=(pow(10,(-100))))){
                t2+=gamma(i,k)*(log(pi(k))-log(gamma(i,k)));
            }
        }
    }
    
    float ELBO_val=t1+t2;
    return ELBO_val;
}

///////////////////////////////////////////////////////////////////////////////////////////////
// Defining functions for K=1

// [[Rcpp::export]]
float grad_theta_weighted_stat_undir_K1(float theta, mat net_adjacency, int N){
    float exp_val=exp(2*theta);
    float exp_val_ratio=exp_val/(1+exp_val);
    float grad_val=0;
    for(int i = 0; i < (N-1); i++){
        for(int j = i+1; j < N; j++){
            bool indicator_0=(net_adjacency(i,j)==0);
            grad_val+=!indicator_0-exp_val_ratio;
        }
    }
    float grad_val_final=2*grad_val;
    return grad_val_final;
}

// [[Rcpp::export]]
float hess_theta_weighted_stat_undir_K1(float theta, int N){
    float exp_val=exp(2*theta);
    float exp_val_ratio=exp_val/(pow((1+exp_val),2));
    float hess_val=0;
    for(int i = 0; i < (N-1); i++){
        for(int j = i+1; j < N; j++){
            hess_val+=(-exp_val_ratio);
        }
    }
    float hess_val_final=4*hess_val;
    return hess_val_final;
}

// [[Rcpp::export]]
float ELBO_conv_weighted_stat_undir_K1(float theta, mat block_dens_mat, mat net_adjacency, int N){
    float exp_val=exp(2*theta);
    float log_exp_val=log(1+exp_val);
    float ELBO_val=0;
    for(int i = 0; i < (N-1); i++){
        for(int j = i+1; j < N; j++){
            int indicator_0=(net_adjacency(i,j)==0);
            if(indicator_0){
                ELBO_val+=(-log_exp_val);
            }else{
                ELBO_val+=(2*theta-log_exp_val+log(block_dens_mat(i,j)));
            }
        }
    }
    return ELBO_val;
}
///////////////////////////////////////////////////////////////////////////////////////////////



