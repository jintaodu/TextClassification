import java.util.Vector;


public class Category {

	/**
	 * @param args
	 */
	public String absolutepath = new String();
	public String categoryname = new String();
	public int categoryid;
	public boolean bLeafCategory = false;
	public Vector<DocNode> docList = new Vector<DocNode>();//此目录下的所有文档节点
	
	public static void main(String[] args) {
		// TODO Auto-generated method stub

	}

}
