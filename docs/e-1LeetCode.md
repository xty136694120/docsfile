# 框架思维

## 数组遍历框架

链表遍历框架， 兼具迭代和递归结构：  

```c++
void traverse(int[] arr) {
    for (int i = 0; i < arr.length; i++) {
    // 迭代访问 arr[i]
    }
}
```

## 链表遍历框架

链表遍历框架， 兼具迭代和递归结构：  

```c++
/* 基本的单链表节点 */
class ListNode {
    int val;
    ListNode next;
} 
void traverse(ListNode head) {
    for (ListNode p = head; p != null; p = p.next) {
    // 迭代访问 p.val
	}
} 
void traverse(ListNode head) {
	// 递归访问 head.val
	traverse(head.next)
}
```

## 树的遍历框架

⼆叉树遍历框架， 典型的⾮线性递归遍历结构：  

```c++
/* 基本的⼆叉树节点 */
class TreeNode {
    int val;
    TreeNode left, right;
} 
void traverse(TreeNode root) {
    traverse(root.left)
    traverse(root.right)
}
```

⼆叉树框架可以扩展为 N 叉树的遍历框架：  

```c++
/* 基本的 N 叉树节点 */
class TreeNode {
    int val;
    TreeNode[] children;
} 
void traverse(TreeNode root) {
    for (TreeNode child : root.children)
    	traverse(child)
}
```

N 叉树的遍历⼜可以扩展为**图**的遍历， 因为图就是好⼏ N 叉棵树的结合
体。 你说图是可能出现**环**的？ 这个很好办， ⽤个布尔数组**visited**做标记就
⾏  



# C++ 刷题tips

## string 转 char

```c++
string s = "ABCDEF";
char a[20];
strcpy(a,s.c_str());
```

## unordered_set

```c++
unordered_set<string> temp;
temp.insert(s.begin(), s.end());

//? temp.count(q);
temp.find(q) != temp.end()
```



