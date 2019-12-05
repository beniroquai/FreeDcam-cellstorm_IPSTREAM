package freed.cam.apis.camera2.modules;

import android.hardware.Camera;
import android.hardware.camera2.CaptureRequest;
import android.os.Build;
import android.os.Handler;
import android.os.StrictMode;
import android.text.TextUtils;

import com.troop.freedcam.R;

import java.io.BufferedReader;
import java.io.File;
import java.io.IOException;
import java.io.OutputStream;
import java.io.PrintWriter;
import java.net.ServerSocket;
import java.net.Socket;

import freed.cam.apis.basecamera.CameraWrapperInterface;
import freed.cam.apis.basecamera.modules.ModuleHandlerAbstract;
import freed.cam.apis.basecamera.parameters.modes.ToneMapChooser;
import freed.cam.apis.camera2.modules.helper.ImageCaptureHolder;
import freed.settings.SettingKeys;
import freed.settings.SettingsManager;
import freed.utils.Log;


import okhttp3.Call;
import okhttp3.Callback;
import okhttp3.MediaType;
import okhttp3.MultipartBody;
import okhttp3.OkHttpClient;
import okhttp3.Request;
import okhttp3.RequestBody;
import okhttp3.Response;

public class CellStormModule extends PictureModuleApi2 {
    // THIS IS THE OLD VERSION WITHOUT GITHUB SUPPORT!

    private final String TAG = CellStormModule.class.getSimpleName();

    private boolean continueCapture = false;
    private final int cropSize = 200;

    String my_server_ip = "192.168.43.86";
    int my_portnumber = 5000;
    Socket mysocket;
    OutputStream myOutputStream;
    PrintWriter myprintwriter;

    public CellStormModule(CameraWrapperInterface cameraUiWrapper, Handler mBackgroundHandler, Handler mainHandler) {
        super(cameraUiWrapper, mBackgroundHandler, mainHandler);
        name = cameraUiWrapper.getResString(R.string.module_cellstorm);
        Log.i(TAG, "This is cellSTORM1!");
    }



    @Override
    public void InitModule() {
        super.InitModule();
        if (cameraUiWrapper.getActivityInterface().getPermissionManager().hasWifiPermission(null))
            connectServer();
        // connect to server for streaming the bytes
    }



    @Override
    public String LongName() {
        return "CellStorm";
    }

    @Override
    public String ShortName() {
        return "Cell";
    }

    @Override
    public void DoWork() {
        Log.i(TAG, "This is cellSTORM!");
        if (continueCapture)
            continueCapture = false;
        else {
            continueCapture = true;
            super.DoWork();
        }
    }

    @Override
    protected void prepareCaptureBuilder(int captureNum) {
        currentCaptureHolder.setCropSize(cropSize, cropSize);
    }

    @Override
    public void internalFireOnWorkDone(File file) {
        fireOnWorkFinish(file);
    }


    public void connectServer() {
        try {
            StrictMode.ThreadPolicy policy = new StrictMode.ThreadPolicy.Builder()
                    .permitAll().build();
            StrictMode.setThreadPolicy(policy);
            //your codes here
            mysocket = new Socket(my_server_ip, my_portnumber);
            myOutputStream= mysocket.getOutputStream();


        } catch (IOException e) {
            e.printStackTrace();
        }
    }

    @Override
    public void captureStillPicture() {

        Log.d(TAG,"########### captureStillPicture ###########");
        currentCaptureHolder = new ImageCaptureHolder(cameraHolder.characteristics, captureDng, captureJpeg, cameraUiWrapper.getActivityInterface(),this,this, this);
        currentCaptureHolder.setFilePath(getFileString(), SettingsManager.getInstance().GetWriteExternal());
        currentCaptureHolder.setForceRawToDng(SettingsManager.get(SettingKeys.forceRawToDng).get());
        currentCaptureHolder.setToneMapProfile(((ToneMapChooser)cameraUiWrapper.getParameterHandler().get(SettingKeys.TONEMAP_SET)).getToneMap());
        currentCaptureHolder.setSupport12bitRaw(SettingsManager.get(SettingKeys.support12bitRaw).get());
        currentCaptureHolder.setOutputStream(myOutputStream);

        Log.d(TAG, "captureStillPicture ImgCount:"+ BurstCounter.getImageCaptured() +  " ImageCaptureHolder Path:" + currentCaptureHolder.getFilepath());

        String cmat = SettingsManager.get(SettingKeys.MATRIX_SET).get();
        if (cmat != null && !TextUtils.isEmpty(cmat) &&!cmat.equals("off")) {
            currentCaptureHolder.setCustomMatrix(SettingsManager.getInstance().getMatrixesMap().get(cmat));
        }

        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.KITKAT) {
            rawReader.setOnImageAvailableListener(currentCaptureHolder,mBackgroundHandler);
        }

        //cameraUiWrapper.captureSessionHandler.StopRepeatingCaptureSession();
        //cameraUiWrapper.captureSessionHandler.CancelRepeatingCaptureSession();
        prepareCaptureBuilder(BurstCounter.getImageCaptured());
        changeCaptureState(ModuleHandlerAbstract.CaptureStates.image_capture_start);
        Log.d(TAG, "StartStillCapture");
        cameraUiWrapper.captureSessionHandler.StartImageCapture(currentCaptureHolder, mBackgroundHandler);
        currentCaptureHolder.save();
    }
}