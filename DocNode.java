import java.util.HashMap;


public class DocNode {

	/**
	 * @param args
	 */
	public int label;
	public String absolutepath;
	public HashMap<String,Integer> tf; //ÎÄµµµÄ´Ê
	public DocNode(int directory,String absopath)
	{
	  	label = directory;
	  	absolutepath = absopath;
	  	tf = new HashMap<String,Integer>();
	}
	public static void main(String[] args) {
		// TODO Auto-generated method stub

	}

}
