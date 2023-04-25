#input
"abcde..."
#output 
'a','b','c'
s="abcdefghiklmnopqr"
output=""
for c in s:
    output+="'"
    output+=c
    output+="',"
print(output)
