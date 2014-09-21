
import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.File;
import java.io.FileInputStream;
import java.io.FileNotFoundException;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.InputStreamReader;
import java.io.OutputStreamWriter;
import java.io.PrintWriter;
import java.text.SimpleDateFormat;
import java.util.Date;
import java.util.HashMap;
import java.util.Vector;

public class HieSVM {

	public static int m;
	// public static int[]
	// a={-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,0,0,0,0,0,0,1,1,1,1,1,1,1,1,2,2,2,2,2,2,2,3,3,3,3,4,4,4,4,5,5,5,5,6,6,6,6,6,7,7,7,8,8,8,8,8,8,9,9,9,10,10,10,10,11,11,11,11,11,11};
	//public static int[] a = { -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,-1,-1,-1,-1,-1};
	//public static int[] a = {-1,0,0,0,0,0,0,-1,7,7,7,7,7,7,7,7,-1,16,16,16,16,-1,21,21,21,-1,25,25,25,-1,29,29,29,29,-1,34,34,34,34,34,34,34,-1,42,42,42,42,-1,47,47,47,47,47,47,-1,54,54,54,54,-1,59,59,59,59,59,-1,65,65,65,65,65,65};
	
	public static int[] a = {-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1};
	// public static int[] a=null;
	public static int[] d = { 6, 8, 7, 4, 4, 4, 5, 3, 6, 3, 4, 6 };
	public static double lambdamin = 1;
	public static int dim;
	private double[][] w;

	public HieSVM(int ml, int diml) {
		m = ml;
		dim = diml;
		w = new double[m][dim];// m为类别数，dim为向量维数
	}

	static void geta() throws Exception {
		int num = 0;
		BufferedReader br = new BufferedReader(new InputStreamReader(
				new FileInputStream("E:\\Parents.txt")));
		String temp = br.readLine();
		while (temp != null) {
			num++;
			temp = br.readLine();
		}
		br.close();
		a = new int[num];
		br = new BufferedReader(new InputStreamReader(new FileInputStream(
				"E:\\Parents.txt")));
		temp = br.readLine();
		int i = 0;
		while (temp != null) {
			a[i++] = Integer.valueOf(temp);
			temp = br.readLine();
		}
		br.close();
	}

	static int parent(int child) throws Exception {

		if (child >= 0 && child < a.length)
			return a[child];
		else
			return -1;

	}

	static int getSib(int index, int[] sibs) throws Exception {
		int snum = 0;
		int i;
		for (i = 0; i < a.length; i++)
			if (i != index && a[i] == a[index])
				sibs[snum++] = i;
		return snum;
	}

	static int getChild(int index, Vector<Integer> childs) throws Exception {
		int cnum = 0;
		int i;
		for (i = 0; i < a.length; i++)
			if (a[i] == index)
			{
				childs.add(i);
				cnum++;
			}
				
		return cnum;
	}

	static void computeK(double[][] K) throws Exception {
		int i, j;
		for (i = 0; i < m; i++)
			for (j = 0; j < m; j++) {
				if (i == j)
					K[i][j] = 1;
				else
					K[i][j] = 0.0;
			}
		double[][] K_up = new double[m][m];
		for (i = 0; i < m; i++)
			for (j = 0; j < m; j++) {
				if (i == j)
					K_up[i][j] = K[i][j];
				else
					K_up[i][j] = -1 * K[i][j];
			}
		PrintWriter pw = new PrintWriter(new OutputStreamWriter(
				new FileOutputStream("E:\\K_up.txt")), true);
		for (i = 0; i < m; i++) {
			for (j = 0; j < m; j++)
				pw.print(K_up[i][j] + " ");
			pw.print("\r\n");
		}
		pw.close();

	}

	static int getdesnum(int i) throws Exception {
		Vector<Integer> subs = new Vector<Integer>();
		int snum = getChild(i, subs);
		if (snum == 0)
			return 1;
		else {
			int total = 0;
			for (int j = 0; j < snum; j++)
				total += getdesnum(subs.get(j));
			return total;
		}
	}

	static int isAncestor(int i, int j) throws Exception {
		int k = a[j];
		while (k != -1) {
			if (k == i)
				return 1;
			k = a[k];
		}
		return 0;
	}

	private void compute_g_sparse(double[][] g, Vector trainfiles, double C,
			double[][] w, double[][] K) throws Exception {// 20121104 train the
															// model with sparse
															// data
		double[][] g_ou = new double[m][dim];// 为omega(w)的梯度
		double[][] g_h = new double[m][dim];// 为H(W)的梯度
		int i, j, k;
		for (i = 0; i < m; i++)
			for (j = 0; j < dim; j++)
				g_h[i][j] = 0;

		for (i = 0; i < m; i++) {
			//System.out.println("i="+i);
			for (k = 0; k < dim; k++) {
				if (w[i][k] > 0) {
					g_ou[i][k] = K[i][i] * w[i][k];
				} else if (w[i][k] < 0) {
					g_ou[i][k] = K[i][i] * w[i][k];
				} else {
					g_ou[i][k] = K[i][i] * w[i][k];
				}

			}// end of for( k = 0
			for (j = 0; j < m; j++)
				if (j != i) {
					double dotmul = 0;// 点乘
					for (k = 0; k < dim; k++)
						dotmul += w[i][k] * w[j][k];
					int sign;
					if (dotmul > 0)
						sign = 1;
					else if (dotmul < 0)
						sign = -1;
					else
						sign = 0;
					for (k = 0; k < dim; k++)
						g_ou[i][k] += sign * K[i][j] * w[j][k];
				}
		}

		int N = trainfiles.size();

		for (k = 0; k < N; k++) {
			//System.out.println("k="+k);
			String[] subs = trainfiles.elementAt(k).toString().split(" ");
			int curdim = subs.length - 1;
			int label = Integer.valueOf(subs[curdim]);
			int[] indexs = new int[curdim];
			double[] x = new double[curdim];
			int l;
			for (l = 0; l < curdim; l++) {
				String[] subs2 = subs[l].split(":");
				indexs[l] = Integer.valueOf(subs2[0]);
				x[l] = Double.valueOf(subs2[1]);
			}
			int maxind_i = -1;
			int maxind_j = -1;
			double max = -1;
			i = label;
			while (i != -1) {
				int[] sibs = new int[100];// 每个结点最多有100个兄弟结点
				int snum = 0;
				snum = getSib(i, sibs);// 得到兄弟结点的数量
				// if(k%100==0)
				// System.out.print(i+" "+snum);
				for (j = 0; j < snum; j++) {
					double obj = 1;

					for (l = 0; l < curdim; l++)
						// 维度数量curdim
						obj += (-1 * w[i][indexs[l]] * x[l] + w[sibs[j]][indexs[l]]
								* x[l]);
					if (max < obj) {
						max = obj;
						maxind_i = i;
						maxind_j = sibs[j];
					}
				}// end of for
				i = a[i];
			}// end of while

			if (max > 0) {
				for (i = 0; i < m; i++)
					for (j = 0; j < curdim; j++)
						if (i == maxind_i)
							g_h[i][indexs[j]] += (-1 * x[j] * Math.exp(max) / (1 + Math
									.exp(max)));
						else if (i == maxind_j)
							g_h[i][indexs[j]] += x[j] * Math.exp(max)
									/ (1 + Math.exp(max));
			}
		}// end of for( k = 0; k < N; k++ )训练集中的每一个样本
		// System.out.println("N="+N);
		// PrintWriter pw = new PrintWriter(new OutputStreamWriter(new
		// FileOutputStream("E:\\g0-sparse.txt")),true);
		for (i = 0; i < m; i++) {
			for (j = 0; j < dim; j++) {
				g_h[i][j] = C / N * g_h[i][j];
				g[i][j] = g_ou[i][j] + g_h[i][j] - lambdamin * w[i][j];
				// g[i][j]=g_ou[i][j]+g_h[i][j];
				// pw.print(g[i][j]+" ");
			}
			// pw.print("\n");
		}
		// pw.close();
	}// end of void compute_g_sparse(double[][]g

	static double compute_sparse(double[][] w) throws Exception {
		int cnt = 0;
		int i, j;
		for (i = 0; i < m; i++)
			for (j = 0; j < dim; j++) {

				if (w[i][j] == 0)
					cnt += 1;
			}
		return (double) cnt / (m * dim);
	}

	public void RDA_large(Vector trainfiles, Vector testfiles, double C,
			double epsilon) throws Exception {// 20121104 for largescale classes
		// Regualrized Dual Averaging Method
		// w为超平面的法向量
		int i, j;
		double[][] g = new double[m][dim];// m个目录，每个目录一个梯度向量
		double[][] g_up = new double[m][dim];
		for (i = 0; i < m; i++)
			for (j = 0; j < dim; j++) {
				w[i][j] = 0;
				g_up[i][j] = 0;
			}
		double[][] K = new double[m][m];
		computeK(K);// 初始化K矩阵为单位矩阵

		File hieSVMmodel = new File(".\\SVMModel\\hieSVMmodel.txt");
		if (hieSVMmodel.exists())
			hieSVMmodel.delete();
		hieSVMmodel.createNewFile();
		BufferedWriter hieSVMmodelbw = new BufferedWriter(
				new OutputStreamWriter(new FileOutputStream(hieSVMmodel)));
		double maxacc = 0;
		SimpleDateFormat df = new SimpleDateFormat("HH:mm:ss");// 设置日期格式

		double[][] wbak = new double[m][dim];// 记录准确率最高时候的w
		for (int t = 1; t < 20; t++)// 迭代次数,暂设为20次
		{
			System.out.println(df.format(new Date()) + " Now itertor time = "
					+ t);
			compute_g_sparse(g, trainfiles, C, w, K);// 计算每个结点的梯度向量

			for (i = 0; i < m; i++)
				// step 1
				for (j = 0; j < dim; j++) {
					g_up[i][j] = (double) (t - 1) / t * g_up[i][j] + (double) 1
							/ t * g[i][j];
				}
			// double[][] w_old=new double[m][dim];
			for (i = 0; i < m; i++)// step 2 l1-norm
			{
				for (j = 0; j < dim; j++) {
					// w_old[i][j]=w[i][j];
					double tp1 = -1 * g_up[i][j] / lambdamin;
					/*
					 * double sign; if(tp1>0) sign=1; else if(tp1<0) sign=-1;
					 * else sign=0; double tp2;
					 * if(Math.abs(tp1)-0.000001/Math.sqrt(t)>0)
					 * tp2=Math.abs(tp1)-0.000001/Math.sqrt(t); else tp2=0;
					 * w[i][j]=signtp2;
					 */
					w[i][j] = tp1;
				}
			}

			double rate = compute_sparse(w);
			System.out.println(df.format(new Date()) + " sparse: " + rate);
			// double acc=test_classify_sparse(t,testfiles, w);
			double acc = test_classify_sparse(testfiles,1);//Test模式

			if (maxacc < acc) {
				maxacc = acc;
				for (i = 0; i < m; i++)// step 2 l1-norm
				{
					for (j = 0; j < dim; j++) {
						wbak[i][j] = w[i][j];
					}
				}
			}
			System.out.println(df.format(new Date())
					+ " The max accuracy now is " + maxacc);

		}// end of for(int t = 1; t < 200; t++ )//迭代次数
		for (i = 0; i < m; i++)// 保存最佳模型的参数
		{
			for (j = 0; j < dim; j++) {
				hieSVMmodelbw.write(wbak[i][j] + " ");
				hieSVMmodelbw.flush();
			}
			hieSVMmodelbw.write("\r\n");
			hieSVMmodelbw.flush();
		}

		hieSVMmodelbw.close();
		System.out.println("Training Process Over~");
		System.out
				.println(df.format(new Date()) + " The accuracy is " + maxacc);

	}

	public boolean classifyfiles(Vector unclassifyfiles,Vector<String> classifyresult) throws Exception
	{

		HashMap<Integer,String> categoryinfo = new HashMap<Integer,String>();

		File f = new File(".\\categoryinfo.txt");
		if(!f.exists()) 
		{
			System.out.println("目录信息文件\"categoryinfo.txt\"找不到");
			return false;
		}
		BufferedReader br = new BufferedReader(new InputStreamReader(new FileInputStream(f),"UTF-8"));
		String line;
		while((line = br.readLine())!=null)
		{
		  String[] info = line.split("::::");
		  categoryinfo.put(Integer.parseInt(info[0]),info[1]);
		}

		int i, j;
		int count = 0;
		int N = unclassifyfiles.size();

		for (i = 0; i < N; i++) {
			String[] subs = unclassifyfiles.elementAt(i).toString().split(" ");
			int curdim = subs.length - 1;

			int[] indexs = new int[curdim];//样本在哪些维度上的特征非零
			double[] xt = new double[curdim];//样本的特征向量
			int l;
			for (l = 0; l < curdim; l++) {
				String[] subs2 = subs[l].split(":");
				indexs[l] = Integer.valueOf(subs2[0]);
				xt[l] = Double.valueOf(subs2[1]);
			}
			int maxind = -1;//祖先节点
			double max = -1;
			int cnum = 0;//孩子节点数量
			Vector<Integer> childs = new Vector<Integer>();
			cnum = getChild(maxind, childs);
			while (cnum != 0) {//对于每一个变量都从父节点开始分类
				//System.out.println("cnum = "+cnum);
				maxind = -1;
				max = -1;
				for (j = 0; j < cnum; j++) {
					double dotmul = 0;
					for (l = 0; l < curdim; l++)
						dotmul += xt[l] * w[childs.get(j)][indexs[l]];
					if (max < dotmul) {
						max = dotmul;
						maxind = childs.get(j);
					}
				}
				childs.clear();
				cnum = getChild(maxind, childs);
			}
			System.out.println("该文档所在类别编号为："+maxind+"，目录名为："+categoryinfo.get(maxind));
			classifyresult.add(categoryinfo.get(maxind));
			
		}//end of for (i = 0; i < N; i++) {
	
		return true;
	}
	public double test_classify_sparse(Vector testfiles, int mode) throws Exception {// todo
		//mode 分为1-->“test”、2-->“multiple/single document classify”
		
		
		if(!(1 == mode || 2 == mode))
		{
			System.err.println("mode = "+mode+" mode 必须为1或者2");
			return 0.0;
		}
		HashMap<Integer,String> categoryinfo = new HashMap<Integer,String>();
		if(2 == mode)
		{
			File f = new File(".\\categoryinfo.txt");
			if(!f.exists()) 
			{
				System.out.println("目录信息文件\"categoryinfo.txt\"找不到");
				return 0.0;
			}
			BufferedReader br = new BufferedReader(new InputStreamReader(new FileInputStream(f),"UTF-8"));
			String line;
			while((line = br.readLine())!=null)
			{
			  String[] info = line.split("::::");
			  categoryinfo.put(Integer.parseInt(info[0]),info[1]);
			}
		}
		int i, j;
		int count = 0;
		int N = testfiles.size();

		for (i = 0; i < N; i++) {
			String[] subs = testfiles.elementAt(i).toString().split(" ");
			int curdim = subs.length - 1;
			int yt = -1;
			if(1 == mode)
			{
				yt = Integer.valueOf(subs[curdim]);//yt为真实的标签,在Evaluation 模式下才使用词变量
			}
			int[] indexs = new int[curdim];//样本在哪些维度上的特征非零
			double[] xt = new double[curdim];//样本的特征向量
			int l;
			for (l = 0; l < curdim; l++) {
				String[] subs2 = subs[l].split(":");
				indexs[l] = Integer.valueOf(subs2[0]);
				xt[l] = Double.valueOf(subs2[1]);
			}
			int maxind = -1;//祖先节点
			double max = -1;
			int cnum = 0;//孩子节点数量
			Vector<Integer> childs = new Vector<Integer>();
			cnum = getChild(maxind, childs);
			while (cnum != 0) {//对于每一个变量都从父节点开始分类
				//System.out.println("cnum = "+cnum);
				maxind = -1;
				max = -1;
				for (j = 0; j < cnum; j++) {
					double dotmul = 0;
					for (l = 0; l < curdim; l++)
						dotmul += xt[l] * w[childs.get(j)][indexs[l]];
					if (max < dotmul) {
						max = dotmul;
						maxind = childs.get(j);
					}
				}
				childs.clear();
				cnum = getChild(maxind, childs);
			}
			if(1 == mode)
			{
				if (maxind == yt)
					count++;
			}else if(2 == mode)
			{
				System.out.println("该文档所在类别编号为："+maxind+"，目录名为："+categoryinfo.get(maxind));
			}
			
		}//end of for (i = 0; i < N; i++) {

		if(1 == mode)
		{
			double acc = (double) count / N;
			System.out.println("分类准确率为："+acc+" ("+count+"/"+N+")");
			return acc;
		}
		return -1.0;//用于查错的小数

	}

	public int ReadNormalVector(int m, int dim) throws IOException {// m表示类别数，dim表示维数
		File hieSVMmodel = new File(".\\SVMModel\\hieSVMmodel.txt");
		if (!hieSVMmodel.exists())
			return -1;// 模型参数文件不存在

		BufferedReader hieSVMmodelbr = new BufferedReader(
				new InputStreamReader(new FileInputStream(hieSVMmodel)));

		String line;
		int count = 0;
		while ((line = hieSVMmodelbr.readLine()) != null) {
			String[] weight = line.trim().split(" ");
			if (weight.length != dim)
				return -2;// 模型维数不符合要求
			for (int i = 0; i < weight.length; ++i)
				w[count][i] = Double.parseDouble(weight[i]);
			count++;
			if (count == dim) {
				line = hieSVMmodelbr.readLine();
			}
		}
		return 1;// 模型读入成功
	}

	public static void main(String[] args) throws Exception {

		//String train_file = "C:\\JAVA WorkSpace\\eclipse 3.4.2 32bits\\ICT\\trainfeaturevectorfile.txt";

		//String test_file = "C:\\JAVA WorkSpace\\eclipse 3.4.2 32bits\\ICT\\testfeaturevectorfile.txt";

		String train_file = "E:\\sogou_sparse_train.txt";
		String test_file = "E:\\sogou_sparse_test.txt";
		BufferedReader br = new BufferedReader(new InputStreamReader(
				new FileInputStream(train_file)));
		String temp = br.readLine();
		Vector trainfiles = new Vector();// 训练文件的特征向量
		while (temp != null) {// 读取训练样本向量表
			trainfiles.add(temp);
			temp = br.readLine();
		}
		br.close();
		br = new BufferedReader(new InputStreamReader(new FileInputStream(
				test_file)));
		temp = br.readLine();
		Vector testfiles = new Vector();// 测试文件的特征向量
		while (temp != null) {// 读取测试样本向量表
			testfiles.add(temp);
			temp = br.readLine();
		}
		br.close();
		testfiles.clear();
		int deletedcount = 0;
		System.out.println("original trainfiles size = " + trainfiles.size());
		for (int i = 0; i < trainfiles.size(); ++i)
			if (i % 3 == 0) {
				testfiles.add(trainfiles.elementAt(i - deletedcount));
				trainfiles.removeElementAt(i - deletedcount);
				deletedcount++;
			}
		System.out.println("final trainfiles size = " + trainfiles.size());
		HieSVM SVM = new HieSVM(10, 10000);
		SVM.RDA_large(trainfiles, testfiles, 1, 0.0001);// Regualrized Dual
														// Averaging Method
		// SVM.ReadNormalVector(15, 10000);
		// SVM.test_classify_sparse(testfiles);
	}
}
