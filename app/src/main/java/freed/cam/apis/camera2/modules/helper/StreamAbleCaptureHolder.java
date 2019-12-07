package freed.cam.apis.camera2.modules.helper;

import android.hardware.camera2.CameraCharacteristics;
import android.media.Image;
import android.os.Build;
import android.support.annotation.RequiresApi;

import java.io.BufferedInputStream;
import java.io.BufferedOutputStream;
import java.io.IOException;
import java.io.OutputStream;
import java.net.Socket;
import java.nio.ByteBuffer;

import freed.ActivityInterface;
import freed.cam.apis.basecamera.modules.ModuleInterface;
import freed.cam.apis.basecamera.modules.WorkFinishEvents;

public class StreamAbleCaptureHolder extends ImageCaptureHolder {

    //the connection to the server
    private Socket socket;
    //used to send data to the target
    private BufferedOutputStream bufferedOutputStream;

    public StreamAbleCaptureHolder(CameraCharacteristics characteristicss, boolean isRawCapture, boolean isJpgCapture, ActivityInterface activitiy, ModuleInterface imageSaver, WorkFinishEvents finish, RdyToSaveImg rdyToSaveImg, Socket socket) {
        super(characteristicss, isRawCapture, isJpgCapture, activitiy, imageSaver, finish, rdyToSaveImg);
        this.socket =socket;
        try {
            this.bufferedOutputStream = new BufferedOutputStream(socket.getOutputStream());
        } catch (IOException e) {
            e.printStackTrace();
        }
    }

    @RequiresApi(api = Build.VERSION_CODES.KITKAT)
    @Override
    public void saveImage(Image image, String f) {
        ByteBuffer buffer = image.getPlanes()[0].getBuffer();
        byte[] bytes = new byte[buffer.remaining()];
        buffer.get(bytes);
        try {
            //sending plain bayer bytearray with simple start end of file
            bufferedOutputStream.write(Byte.parseByte("start"));
            bufferedOutputStream.write(bytes);
            bufferedOutputStream.write(Byte.parseByte("end"));
            bufferedOutputStream.flush();
        } catch (IOException e) {
            e.printStackTrace();
        }
    }
}
