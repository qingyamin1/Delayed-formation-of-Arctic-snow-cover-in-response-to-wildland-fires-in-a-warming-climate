setwd("/home/qingyamin/Downloads/code/code")  # Working directory
rm(list=ls())                                # Clean workspace

library(svglite)
require(signal)
require(burnr)
require(binhf)

# ======= 内嵌 SEA 主函数 =======
sea_dbl <- function(events, y, yr, preyr, postyr, dbl, nboot, nboot_event,
                    nsample, graphics, figtype, bootmethod, proxytype){
  if(missing(proxytype)) {proxytype<- 'temperature'}
  if(missing(bootmethod)) {bootmethod<- 'random'}
  if(missing(figtype)) {figtype<- 'series'}
  if(missing(graphics)) {graphics<- T}
  if(missing(nsample)) {nsample <- 0}
  if(missing(nboot_event)) {nboot_event <- 0}
  if(missing(nboot)) {nboot <- 100}
  if(missing(dbl))   {dbl<- F}
  if(missing(postyr)){postyr<- 5}
  if(missing(preyr)) {preyr<- 5}
  
  # Excise events if their window exceeds time series range
  if (any(events > (max(yr)-postyr)) || any(events < (min(yr)+preyr)) && dbl == 1){
    print('Warning: excising event because its window exceeds the length of the time series!')
    rm1 <- which(events < (min(yr)+preyr))
    rm2 <- which(events > (max(yr)-postyr))
    events <- events[-c(rm1,rm2)]
  }
  
  strt_index <- which(yr==(min(events)-preyr))
  stop_index <- which(yr==max(yr))
  y <-  y[strt_index:stop_index]
  yr<- yr[strt_index:stop_index]
  remove (strt_index, stop_index)
  
  nevents = length(events)
  
  # Total number of possible unique draws
  if (dbl==0 || dbl==F) {
    nsample     <- 0
    nboot_event <-0
  } else {
    total_draws <- (factorial(nevents)/(factorial(nsample)*factorial(nevents-nsample)));
    print(c('Total possible unique key year draws:', total_draws))
    if (total_draws < nboot_event) {
      print('¡Reduce total number of key year draws!')
      print('******')
      print('SEA with no confidence intervals on response')
      print('******')
      dbl <- 0
      nboot_event  <- 0
      nsample <- 0
    }
    remove (total_draws)
  }
  
  if (dbl==0 || dbl==F) {
    seaM <- matrix(NA, nrow = nevents, ncol = (postyr+preyr+1))
  } else {
    seaM <- array(NA, c(nsample, (postyr+preyr+1), nboot_event))
  }
  
  if (dbl==0 || dbl==F) {
    for (i in 1:nevents){
      centerYear <- which(yr == events[i])
      seaM[i,] <- y[seq((centerYear-preyr),(centerYear+postyr))]
    }
  } else {
    draws <- matrix(NA, nrow = nboot_event, ncol =  nsample)
    for (i in 1:nboot_event){
      draws[i,] <- sort(sample(events, nsample, replace=FALSE))
      j=0
      while (anyDuplicated(draws[1:i,]) != 0){
        draws[i,] <- sort(sample(events, nsample, replace=FALSE))
        j <- j+1
        if (j==100){
          print('100 iterations without a unique key year draw')
          print('Reduce draw size relative to total number of events')
        }
      }
    }
    remove(j)
    for (j in 1:nboot_event){
      sample_yrs <- draws[j,]
      for (k in 1:nsample){
        centerYear <- which(yr == sample_yrs[k])
        seaM[k,,j] <- y[seq((centerYear-preyr),(centerYear+postyr))]
      }
      remove(sample_yrs)
    }
  }
  
  seaMoriginal <- seaM
  
  if  (dbl==0 || dbl==F){
    seaM <- apply(seaM,2, function(x) x-rowMeans(seaM[,(1:preyr)]))
    nrows     <- dim(seaM)[1]
    ncolumns  <- dim(seaM)[2]
    sval    <- colMeans(seaM, na.rm = TRUE)
    seaSD   <- apply(seaM,2,sd, na.rm = TRUE)
  } else {
    mean_resp = matrix(NA, nrow=nboot_event, ncol = (preyr+1+postyr))
    for (n in 1:nboot_event){
      mean_resp[n,] <- colMeans(apply(seaM[,,n],2, function(x) x-rowMeans(seaM[,(1:preyr),n])))
    }
    nrows   <- dim(seaMoriginal)[1]
    ncolumns<- dim(seaMoriginal)[2]
    sval <- apply(mean_resp,2,function (x) quantile(x, c(.05,.50, .95)))
  }
  
  if (bootmethod=='random'){
    rseaMbar = matrix(NA,nrow = nboot,ncol = ncolumns)
    yr_temp <- yr[(preyr+1):(length(yr)-postyr)]
    y_temp  <-  y[(preyr+1):(length(yr)-postyr)]
    for (nmct in 1:nboot){
      if (dbl==0 || dbl==F){
        randomEvents <- sort(sample(yr_temp, size = nevents, replace = FALSE))
      } else {
        randomEvents <- sort(sample(yr_temp, size = nsample, replace = FALSE))
      }
      rseaM <- matrix(NA, length(randomEvents),postyr+preyr+1)
      for (i in 1:length(randomEvents)){
        rcenterYear <- which(yr == randomEvents[i])
        rseaM[i,]   <- y[seq((rcenterYear-preyr),(rcenterYear+postyr))]
      }
      rseaM <- apply(rseaM,2, function (x) x-rowMeans(rseaM[,c(1:preyr)]))
      rseaMbar[nmct,]  <- colMeans(rseaM)
    }
    remove('yr_temp', 'y_temp')
    sci <- apply(rseaMbar,2,function (x) quantile(x, c(0.01,0.05,0.10,0.90,0.95, 0.99)))
  }
  
  syr <- seq(-preyr,postyr,by=1)
  
  if ((figtype=='series') && (graphics ==1 || graphics ==T)){
    par(mfrow=c(1,1))
    if (dbl==1 || dbl==T) {
      plot(syr,sval[2,], lwd=2, type='l',col='darkred', las=1,
           xlim= c(preyr*-1, postyr),
           xlab= c(''), ylab= c(""), ylim=c(-1.5,1.5))
      polygon(c(syr, rev(syr)), c(sval[3,], rev(sval[1,])), col = "azure3", border = NA)
      lines(syr,sval[2,], lwd=2, type='l',col='darkred')
      points(syr,sval[2,], col="black", pch=3,cex=1)
    } else {
      plot(syr,sval, lwd=2, type='l',col='darkred', las=1,
           xlim= c(preyr*-1, postyr), xlab= c(''), ylab= c(""),
           ylim=c(-round(max(abs(sval)),1),round(max(abs(sval)),digits=1)))
      points(syr,sval, col="black", pch=3,cex=1)
    }
    for (i in 2:5){
      lines(syr, sci[i,],lty=2,col='black',lwd=1)
    }
    abline(h=0, lwd=1, lty=2, col='black')
    abline(v=0, lwd=1, lty=2, col='black')
  }
  if (dbl==1 || dbl==T){colnames(sval) <- syr}
  colnames(sci)  <- syr
  list_out <- list(sval,syr,sci,seaM,seaMoriginal)
  data.names <- c("sval","syr","sci","seaM","seaMoriginal")
  names(list_out) <- paste(data.names)
  return(list_out)
}

# ========= 你的数据和参数设置 =========
data1 <- read.csv('../Arctic-BA-Start-day-1982-2018-Fig-4-data.csv',header=TRUE)
fire <- data1[,2]



mean_val <- mean(fire)
std <- sd(fire)
threshold <- mean_val + 1*std
event <- rep(0, length(fire))
for(m in 1:length(fire)){
  if(fire[m] >= threshold){
    event[m] = m
  }
}
event[event == 0] <- NA
event <- na.omit(event)
events <- as.numeric(event)

y  <- data1[1:(nrow(data1)-3),3]
yr <- data1[1:(nrow(data1)-3),1]
preyr  <- 3
postyr <- 3
dbl   <- TRUE
nboot <- 1000
nboot_event <- 1000
nsample<- 1
bootmethod <- 'random'
graphics  <- TRUE
figtype   <- 'series'
proxytype <- 'precipitation'

sea_out <- sea_dbl(events, y, yr, preyr, postyr, dbl, nboot, nboot_event,
                   nsample, graphics, figtype, bootmethod, proxytype)

# 结果输出
print("Composite response:")
print(sea_out$sval)
print("Window (years):")
print(sea_out$syr)
print("SCI Confidence Intervals:")
print(sea_out$sci)
