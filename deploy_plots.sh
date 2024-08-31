if test -d $WWW/plots/$1; then
	echo "$WWW/plots/$1 directory exists."
  	exit 0;
fi


pb_deploy_plots.py figures/$1 $WWW/plots -c -j 16 -r
cp out/$1.root $WWW/plots/$1
