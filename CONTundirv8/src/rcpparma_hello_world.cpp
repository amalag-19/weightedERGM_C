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

///////////////////////////////////////////////////////////////////////////////////////////////////////////////
// gamma, Tau update function, gradient and hessian functions and ELBO convergence function

// [[Rcpp::export]]
cube gamma_update_weighted_stat_undir_parametric_gamma(mat gamma, vec pi, vec theta, mat alpha_mat, mat beta_mat, mat log_Gamma_alpha_mat, mat net_adjacency, mat net_weight, int N, int K){
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
                            t1+=(gamma(j,l)/(2*gamma(i,k)))*(log(exp_val_ratio_mat(k,l))+alpha_mat(min(k,l),max(k,l))*log(beta_mat(min(k,l),max(k,l)))-log_Gamma_alpha_mat(min(k,l),max(k,l))+log(net_weight(i,j))*(alpha_mat(min(k,l),max(k,l))-1)-net_weight(i,j)*beta_mat(min(k,l),max(k,l)));
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
mat grad_alpha_mat_weighted_stat_undir_parametric_gamma(mat gamma, mat beta_mat, mat Digamma_alpha_mat, mat net_adjacency, mat net_weight, int N, int K){
    mat grad_alpha_mat(K,K,fill::zeros);
    for(int i = 0; i < (N-1); i++){
        for(int j = i+1; j < N; j++){
            mat grad_alpha_sub_mat(K,K,fill::zeros);
            for(int k = 0; k < K; k++){
                for(int l = 0; l < K; l++){
                    int indicator_0=(net_adjacency(i,j)==0);
                    if(!indicator_0){
                        grad_alpha_sub_mat(k,l)=gamma(i,k)*gamma(j,l)*(log(beta_mat(min(k,l),max(k,l)))-Digamma_alpha_mat(min(k,l),max(k,l))+(log(net_weight(i,j))));
                    }
                }
            }
            grad_alpha_mat+=grad_alpha_sub_mat;
        }
    }
    return grad_alpha_mat;
}

// [[Rcpp::export]]
mat hess_alpha_mat_weighted_stat_undir_parametric_gamma(mat gamma, mat Trigamma_alpha_mat, mat net_adjacency, int N, int K){
    mat hess_alpha_mat(K,K,fill::zeros);
    for(int i = 0; i < (N-1); i++){
        for(int j = i+1; j < N; j++){
            mat hess_alpha_sub_mat(K,K,fill::zeros);
            for(int k = 0; k < K; k++){
                for(int l = 0; l < K; l++){
                    int indicator_0=(net_adjacency(i,j)==0);
                    if(!indicator_0){
                        hess_alpha_sub_mat(k,l)=gamma(i,k)*gamma(j,l)*(-Trigamma_alpha_mat(min(k,l),max(k,l)));
                    }
                }
            }
            hess_alpha_mat+=hess_alpha_sub_mat;
        }
    }
    return hess_alpha_mat;
}

// [[Rcpp::export]]
mat grad_beta_mat_weighted_stat_undir_parametric_gamma(mat gamma, mat alpha_mat, mat beta_mat, mat net_adjacency, mat net_weight, int N, int K){
    mat grad_beta_mat(K,K,fill::zeros);
    for(int i = 0; i < (N-1); i++){
        for(int j = i+1; j < N; j++){
            mat grad_beta_sub_mat(K,K,fill::zeros);
            for(int k = 0; k < K; k++){
                for(int l = 0; l < K; l++){
                    int indicator_0=(net_adjacency(i,j)==0);
                    if(!indicator_0){
                        grad_beta_sub_mat(k,l)=gamma(i,k)*gamma(j,l)*((alpha_mat(min(k,l),max(k,l))/beta_mat(min(k,l),max(k,l)))-net_weight(i,j));
                    }
                }
            }
            grad_beta_mat+=grad_beta_sub_mat;
        }
    }
    return grad_beta_mat;
}

// [[Rcpp::export]]
mat hess_beta_mat_weighted_stat_undir_parametric_gamma(mat gamma, mat alpha_mat, mat beta_mat, mat net_adjacency, int N, int K){
    mat hess_beta_mat(K,K,fill::zeros);
    for(int i = 0; i < (N-1); i++){
        for(int j = i+1; j < N; j++){
            mat hess_beta_sub_mat(K,K,fill::zeros);
            for(int k = 0; k < K; k++){
                for(int l = 0; l < K; l++){
                    int indicator_0=(net_adjacency(i,j)==0);
                    if(!indicator_0){
                        hess_beta_sub_mat(k,l)=gamma(i,k)*gamma(j,l)*(-alpha_mat(min(k,l),max(k,l))/(pow(beta_mat(min(k,l),max(k,l)),2)));
                    }
                }
            }
            hess_beta_mat+=hess_beta_sub_mat;
        }
    }
    return hess_beta_mat;
}

// [[Rcpp::export]]
float ELBO_conv_weighted_stat_undir_parametric_gamma(mat gamma, vec pi, vec theta, mat alpha_mat, mat beta_mat, mat log_Gamma_alpha_mat, mat net_adjacency, mat net_weight, int N, int K){
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
                        t1+=(gamma(i,k)*gamma(j,l)*(log(exp_val_ratio_mat(k,l))+alpha_mat(min(k,l),max(k,l))*log(beta_mat(min(k,l),max(k,l)))-log_Gamma_alpha_mat(min(k,l),max(k,l))+log(net_weight(i,j))*(alpha_mat(min(k,l),max(k,l))-1)-net_weight(i,j)*beta_mat(min(k,l),max(k,l))));
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
/*
 // [[Rcpp::export]]
 float grad_HMM_stat_undir_K1(float theta, mat network, int N){
 float exp_val=exp(2*theta);
 float exp_val_ratio=exp_val/(1+exp_val);
 float grad_val=0;
 for(int i = 0; i < (N-1); i++){
 for(int j = i+1; j < N; j++){
 grad_val+=network(i,j)-exp_val_ratio;
 }
 }
 float grad_val_final=2*grad_val;
 return grad_val_final;
 }
 
 // [[Rcpp::export]]
 float hess_HMM_stat_undir_K1(float theta, int N){
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
 float ELBO_conv_HMM_stat_undir_K1(float theta, mat network, int N){
 float exp_val=exp(2*theta);
 float log_exp_val=log(1+exp_val);
 float ELBO_val=0;
 for(int i = 0; i < (N-1); i++){
 for(int j = i+1; j < N; j++){
 ELBO_val+=(network(i,j)*(2*theta)-log_exp_val);
 }
 }
 return ELBO_val;
 }*/
///////////////////////////////////////////////////////////////////////////////////////////////

