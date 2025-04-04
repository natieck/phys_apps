function calc_convolution_weight(nfilter,norder,nder,xdata,xsum,ip)
{
	let i,j,k;
	let w1,w2,w3,w4;
    let am=[],bv=[],gv=[],pm=[],hm=[];

    for(i=0;i<=norder;i++){
        am[i]=[];
        pm[i]=[];
        hm[i]=[];
    }

    if(nfilter==1||norder==0){
		for(i=1;i<=nfilter;i++)hm[norder][i]=1.0/nfilter;
		return hm;
	}

	for(k=0;k<=norder;k++){
		for(j=0;j<=norder;j++)am[k][j]=0.0;
		am[k][k]=1.0;
	}
	for(i=1;i<=nfilter;i++)pm[0][i]=1.0;
	if(nder)for(i=1;i<=nfilter;i++)hm[0][i]=0.0;

	// the cases of k=0 and k=1
	gv[0]=nfilter;
	bv[0]=xsum[1]/nfilter;
	am[1][0]=-bv[0];
	gv[1]=xsum[2]-bv[0]*xsum[1];
	bv[1]=(xsum[3]-bv[0]*xsum[2])/gv[1]-bv[0];
	for(i=1;i<=nfilter;i++)pm[1][i]=xdata[i]-bv[0];
	if(nder){
		for(i=1;i<=nfilter;i++)hm[0][i]=0.0;
		if(nder==1)for(i=1;i<=nfilter;i++)hm[1][i]=pm[1][i]/gv[1];
		else for(i=1;i<=nfilter;i++)hm[1][i]=0.0;
	}else{
		for(i=1;i<=nfilter;i++)hm[0][i]=1.0/nfilter;
		w1=pm[1][ip]/gv[1];
		for(i=1;i<=nfilter;i++)hm[1][i]=hm[0][i]+pm[1][i]*w1;
	}

	for(k=2;k<=norder;k++){
		w1=bv[k-1];w2=gv[k-1]/gv[k-2];
		am[k][0]=-w1*am[k-1][0]-w2*am[k-2][0];
		for(j=1;j<k;j++)am[k][j]=am[k-1][j-1]-w1*am[k-1][j]-w2*am[k-2][j];
		w1=w2=0.0;
		for(j=0;j<=k;j++){
			w3=am[k][j];
			w1+=w3*xsum[j+k];
			w2+=w3*xsum[j+k+1];
		}
		gv[k]=w1;
		if(w1>1.0e-10){
            bv[k]=w2/w1+am[k][k-1];
            for(i=1;i<=nfilter;i++){
	    		w1=0.0;
		    	w2=xdata[i];
			    w3=1.0;
    			for(j=0;j<=k;j++){
    				w1+=am[k][j]*w3;
    				w3*=w2;
    			}
	    		pm[k][i]=w1;
    		}
		    if(nder){
		    	w1=0.0;
		    	w2=xdata[ip];
		    	w3=1.0;
		    	for(j=nder;j<=k;j++){
			    	w4=1.0;
			    	for(i=j;i>j-nder;i--)w4*=i;
			    	w1+=w4*am[k][j]*w3;
			    	w3*=w2;
		    	}
		    	w1/=gv[k];
		    }else w1=pm[k][ip]/gv[k];
            for(i=1;i<=nfilter;i++)hm[k][i]=hm[k-1][i]+pm[k][i]*w1;
        }else{
            for(i=1;i<=nfilter;i++)hm[norder][i]=hm[k-1][i];
            break;
        }
	}
    return hm;
}

function general_least_squares_smoothing(xdata,ydata,startline,endline,ny,iyv,nfilter,norder,nder,flag,str="glss_")
{
	let i,j,k,l,lw,m,n,nop,indx,ip,ir;
	let xld=[],yld=[],xldp=[],yldp=[],xsum=[],hm,x0,w;
	let yres=[];

	lw=endline-startline+1;
    m=math.floor(nfilter/2);
    nop=2*norder+1;

    for(i=0;i<ny;i++){
        yld[i]=[];
        yldp[i]=[];
    }

	for(j=0;j<ny;j++)yres[j]=[];

	xsum[0]=0.0;

    // read the first local block with nfilter*ncolumn and calculate the sums of the powers of x
	xld[0]=xdata[0];
	x0=xdata[startline];
	xld[1]=0.0;
	for(i=2;i<=nfilter;i++)xld[i]=xdata[startline+i-1]-x0;

	for(j=0;j<ny;j++){
		yld[j][0]=yres[j][0]=str+ydata[iyv[j]][0];
		yld[j][1]=ydata[iyv[j]][startline];
	}
	for(j=0;j<ny;j++)for(i=2;i<=nfilter;i++)yld[j][i]=ydata[iyv[j]][startline+i-1];

	if(flag){
		for(i=1;i<=nop;i++)xsum[i]=math.pow(xld[1],i);
	}else{
		for(i=1;i<=nop;i++){
			xsum[i]=math.pow(xld[1],i);
			for(j=2;j<=nfilter;j++)xsum[i]+=math.pow(xld[j],i);
		}
	}

	// smoothing for the first m+1 blocks
	if(flag){
		for(ip=1;ip<=m+1;ip++){
			n=2*(ip-1)+1;
			hm=calc_convolution_weight(n,norder,nder,xld,xsum,ip);
			if(ip<m+1)for(j=n+1;j<=n+2;j++){
				w=xld[j];
				for(i=1;i<=nop;i++)xsum[i]+=math.pow(w,i);
			}
			for(j=0;j<ny;j++){
				w=0.0;
				for(i=1;i<=n;i++){
					w+=hm[norder][i]*yld[j][i];
				}
				yres[j][ip]=w;
			}
		}
	}else{
		for(ip=1;ip<=m+1;ip++){
			hm=calc_convolution_weight(nfilter,norder,nder,xld,xsum,ip);	
			for(j=0;j<ny;j++){
				w=0.0;
				for(i=1;i<=nfilter;i++)w+=hm[norder][i]*yld[j][i];
				yres[j][ip]=w;
			}
		}
	}

	indx=1;
	ir=startline+nfilter-1;
	for(l=m+2;l<=lw-m;l++){
		ir++;
		w=xld[indx];
		for(i=1;i<=nop;i++)xsum[i]-=math.pow(w,i);
		xld[indx]=w=xdata[ir]-x0;
		for(i=1;i<=nop;i++)xsum[i]+=math.pow(w,i);
		for(j=0;j<ny;j++)yld[j][indx]=ydata[iyv[j]][ir];

		hm=calc_convolution_weight(nfilter,norder,nder,xld,xsum,ip);

		for(j=0;j<ny;j++){
			w=0.0;
			for(i=1;i<=nfilter;i++)w+=hm[norder][i]*yld[j][i];
			yres[j][l]=w;
		}

		if(++indx>nfilter)indx=1;
		if(++ip>nfilter)ip=1;
	}

	if(flag){
		for(l=1;l<=m;l++){
			n=2*(m-l)+1;
			for(j=1;j<=2;j++){
				w=xld[indx];
				for(i=1;i<=nop;i++)xsum[i]-=math.pow(w,i);
				if(++indx>nfilter)indx=1;
			}
			k=indx;
			for(i=1;i<=n;i++){
				xldp[i]=xld[k];
				for(j=0;j<ny;j++)yldp[j][i]=yld[j][k];
				if(++k>nfilter)k=1;
			}
			hm=calc_convolution_weight(n,norder,nder,xldp,xsum,m-l+1);
			for(j=0;j<ny;j++){
				w=0.0;
				for(i=1;i<=n;i++){
					w+=hm[norder][i]*yldp[j][i];
				}
				yres[j][lw-m+l]=w;
			}
		}
	}else{
		for(l=1;l<=m;l++){
			hm=calc_convolution_weight(nfilter,norder,nder,xld,xsum,ip);
			for(j=0;j<ny;j++){
				w=0.0;
				for(i=1;i<=nfilter;i++)w+=hm[norder][i]*yld[j][i];
				yres[j][lw-m+l]=w;
			}
			if(++ip>nfilter)ip=1;
		}
	}
    return yres;
}