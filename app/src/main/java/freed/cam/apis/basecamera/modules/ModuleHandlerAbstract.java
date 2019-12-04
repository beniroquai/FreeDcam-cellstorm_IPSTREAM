/*
 *
 *     Copyright (C) 2015 Ingo Fuchs
 *     This program is free software; you can redistribute it and/or modify
 *     it under the terms of the GNU General Public License as published by
 *     the Free Software Foundation; either version 2 of the License, or
 *     (at your option) any later version.
 *
 *     This program is distributed in the hope that it will be useful,
 *     but WITHOUT ANY WARRANTY; without even the implied warranty of
 *     MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 *     GNU General Public License for more details.
 *
 *     You should have received a copy of the GNU General Public License along
 *     with this program; if not, write to the Free Software Foundation, Inc.,
 *     51 Franklin Street, Fifth Floor, Boston, MA 02110-1301 USA.
 * /
 */

package freed.cam.apis.basecamera.modules;

import android.os.Handler;
import android.os.Looper;
import android.os.Message;
import android.support.annotation.Nullable;

import com.troop.freedcam.R;

import java.lang.ref.WeakReference;
import java.util.AbstractMap;
import java.util.ArrayList;
import java.util.HashMap;

import freed.cam.apis.basecamera.CameraWrapperInterface;
import freed.utils.BackgroundHandlerThread;
import freed.utils.Log;

/**
 * Created by troop on 09.12.2014.
 */
public abstract class ModuleHandlerAbstract implements ModuleHandlerInterface
{
    public enum CaptureStates
    {
        video_recording_stop,
        video_recording_start,
        image_capture_stop,
        image_capture_start,
        continouse_capture_start,
        continouse_capture_stop,
        continouse_capture_work_start,
        continouse_capture_work_stop,
        cont_capture_stop_while_working,
        cont_capture_stop_while_notworking,
        selftimerstart,
        selftimerstop
    }

    public interface CaptureStateChanged
    {
        void onCaptureStateChanged(CaptureStates captureStates);
    }

    private final ArrayList<CaptureStateChanged> onCaptureStateChangedListners;

    private final String TAG = ModuleHandlerAbstract.class.getSimpleName();
    public AbstractMap<String, ModuleInterface> moduleList;
    protected ModuleInterface currentModule;
    protected CameraWrapperInterface cameraUiWrapper;

    protected CaptureStateChanged workerListner;

    //holds all listner for the modulechanged event
    private final ArrayList<ModuleChangedEvent> moduleChangedListner;

    private BackgroundHandlerThread backgroundHandlerThread;

    protected Handler mBackgroundHandler;
    protected Handler mainHandler;

    public ModuleHandlerAbstract(CameraWrapperInterface cameraUiWrapper)
    {
        this.cameraUiWrapper = cameraUiWrapper;
        moduleList = new HashMap<>();
        moduleChangedListner = new ArrayList<>();
        onCaptureStateChangedListners = new ArrayList<>();
        mainHandler = new UiHandler(Looper.getMainLooper(),moduleChangedListner,onCaptureStateChangedListners);
        backgroundHandlerThread = new BackgroundHandlerThread(TAG);
        backgroundHandlerThread.create();
        mBackgroundHandler = new Handler(backgroundHandlerThread.getThread().getLooper());

        workerListner = captureStates -> {
            for (int i = 0; i < onCaptureStateChangedListners.size(); i++)
            {

                if (onCaptureStateChangedListners.get(i) == null) {
                    onCaptureStateChangedListners.remove(i);
                    i--;
                }
                else
                {
                    mainHandler.obtainMessage(UiHandler.CAPTURE_STATE_CHANGED,i,0,captureStates).sendToTarget();
                }
            }
        };
    }

    public void changeCaptureState(CaptureStates states)
    {
        workerListner.onCaptureStateChanged(states);
    }

    /**
     * Load the new module
     * @param name of the module to load
     */
    @Override
    public void setModule(String name) {
        if (currentModule !=null) {
            currentModule.DestroyModule();
            currentModule.SetCaptureStateChangedListner(null);
            currentModule = null;
        }
        currentModule = moduleList.get(name);
        if (currentModule == null)
            currentModule = moduleList.get(cameraUiWrapper.getResString(R.string.module_picture));
        currentModule.InitModule();
        ModuleHasChanged(currentModule.ModuleName());
        currentModule.SetCaptureStateChangedListner(workerListner);
        Log.d(TAG, "Set Module to " + name);
    }

    @Override
    public String getCurrentModuleName() {
        if (currentModule != null)
            return currentModule.ModuleName();
        else return cameraUiWrapper.getResString(R.string.module_picture);
    }

    @Override
    public @Nullable ModuleInterface getCurrentModule() {
        if (currentModule != null)
            return currentModule;
        return null;
    }

    @Override
    public boolean startWork() {
        if (currentModule != null) {
            currentModule.DoWork();
            return true;
        }
        else
            return false;
    }

    @Override
    public void setWorkListner(CaptureStateChanged workerListner)
    {
        if (!onCaptureStateChangedListners.contains(workerListner))
            onCaptureStateChangedListners.add(workerListner);
    }


    public void CLEARWORKERLISTNER()
    {
        if (onCaptureStateChangedListners != null)
            onCaptureStateChangedListners.clear();
    }

    /**
     * Add a listner for Moudlechanged events
     * @param listner the listner for the event
     */
    public  void addListner(ModuleChangedEvent listner)
    {
        if (!moduleChangedListner.contains(listner))
            moduleChangedListner.add(listner);
    }

    /**
     * Gets thrown when the module has changed
     * @param module the new module that gets loaded
     */
    public void ModuleHasChanged(final String module)
    {
        if (moduleChangedListner.size() == 0)
            return;
        for (int i = 0; i < moduleChangedListner.size(); i++)
        {
            if (moduleChangedListner.get(i) == null) {
                moduleChangedListner.remove(i);
                i--;
            }
            else
            {
                mainHandler.obtainMessage(UiHandler.MODULE_CHANGED,i,0, module).sendToTarget();
            }
        }
    }

    //clears all listner this happens when the camera gets destroyed
    public void CLEAR()
    {
        moduleChangedListner.clear();
        backgroundHandlerThread.destroy();
    }


    private static class UiHandler extends Handler
    {
        public static final int MODULE_CHANGED= 0;
        public static final int CAPTURE_STATE_CHANGED = 1;
        WeakReference<ArrayList<ModuleChangedEvent>> weakReferenceModuleChangedListners;
        WeakReference<ArrayList<CaptureStateChanged>> weakReferenceCaptureStateChanged;

        public UiHandler(Looper mainLooper,ArrayList<ModuleChangedEvent> moduleChangedListner,ArrayList<CaptureStateChanged> captureStateChanged) {
            super(mainLooper);
            weakReferenceModuleChangedListners = new WeakReference<>(moduleChangedListner);
            weakReferenceCaptureStateChanged = new WeakReference<>(captureStateChanged);
        }

        @Override
        public void handleMessage(Message msg)
        {
            switch (msg.what) {
                case MODULE_CHANGED:
                    ArrayList<ModuleChangedEvent> moduleChangedListners = weakReferenceModuleChangedListners.get();
                    if (moduleChangedListners != null){
                        if (moduleChangedListners.size() > 0) {
                            ModuleChangedEvent event = moduleChangedListners.get(msg.arg1);
                            if (event != null)
                                event.onModuleChanged((String) msg.obj);
                            else
                                moduleChangedListners.remove(event);
                        }
                    }
                break;
                case CAPTURE_STATE_CHANGED:
                    ArrayList<CaptureStateChanged> onCaptureStateChangedListner = weakReferenceCaptureStateChanged.get();
                if (onCaptureStateChangedListner != null)
                    onCaptureStateChangedListner.get(msg.arg1).onCaptureStateChanged((CaptureStates)msg.obj);
                    break;
                default:
                    super.handleMessage(msg);
            }
        }
    }

}
