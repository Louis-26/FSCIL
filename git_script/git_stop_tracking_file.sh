# uncomment if it is in linux, and need to convert dos to unix
#sed -i 's/\r$//' git_script/git_push.sh
read -p "Enter the name of file you want to stop tracking: " FILE_NAME

git rm --cached $FILE_NAME
